import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from filelock import FileLock, Timeout
from project_state import Project
import download_pexels as dp
from diagnose_tts import inspect_saved_config


class StateTests(unittest.TestCase):
    def test_only_completed_intact_files_are_reused(self):
        with tempfile.TemporaryDirectory() as root:
            project = Project(Path(root), 'test')
            stage = project.stage('audio', 1, {'text': 'hello'})
            stage.start()
            self.assertIsNone(stage.cached())
            audio = stage.path / 'audio.wav'
            audio.write_bytes(b'complete')
            stage.finish({'audio': 'audio.wav'}, [audio])
            self.assertIsNotNone(stage.cached())
            audio.write_bytes(b'corrupt!')  # Same length; hash must detect it.
            self.assertIsNone(stage.cached())
            audio.unlink()
            self.assertIsNone(stage.cached())
            stage.manifest.write_text('{broken')
            self.assertIsNone(stage.cached())

    def test_project_lock_blocks_second_writer(self):
        with tempfile.TemporaryDirectory() as root:
            name = str(Path(root) / '.run.lock')
            with FileLock(name):
                with self.assertRaises(Timeout):
                    with FileLock(name, timeout=0):
                        pass

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                Project(Path(root), '../escape')

    def test_config_diagnostic_does_not_claim_running_model(self):
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / 'config.yaml'
            config.write_text('model:\n  repo_id: chatterbox-turbo\n')
            messages = inspect_saved_config(config, 'ru')
            self.assertIn('NOT confirmation', messages[0])
            self.assertIn('WARNING', messages[1])


class ResumeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [patch.object(dp, 'PROJECT', Project(self.root, 'test')),
                        patch.object(dp, 'MEDIA_TYPE', 'video'),
                        patch.object(dp, 'MEDIA_DOWNLOAD_ENABLED', True),
                        patch.object(dp, 'TTS_SEMAPHORE', asyncio.Semaphore(1))]
        for item in self.patches:
            item.start()
        self.visual = patch.object(dp, '_prepare_visual_uncached', AsyncMock(return_value=('city', None)))
        self.visual_mock = self.visual.start()
        self.audio = patch.object(dp, 'generate_chunk_audio', AsyncMock(side_effect=self.make_audio))
        self.audio_mock = self.audio.start()
        self.duration = patch.object(dp, 'resolve_audio_duration', AsyncMock(return_value=2.0))
        self.duration.start()

    async def asyncTearDown(self):
        self.duration.stop()
        self.audio.stop()
        self.visual.stop()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    async def make_audio(self, index, total, text, client, output, sem, prompt=None, brief=None):
        path = dp.build_audio_path(index, output)
        path.write_bytes(b'test audio')
        return dp.ChunkResult(index, text, prompt, path, 2.0, brief)

    async def run_chunk(self, text='hello'):
        return await dp.process_chunk(1, 1, text, None, self.root, asyncio.Semaphore(4))

    async def test_resume_and_visual_change_preserve_audio(self):
        first = await self.run_chunk()
        second = await self.run_chunk()
        self.assertEqual(first.audio_path, second.audio_path)
        self.assertEqual(self.audio_mock.await_count, 1)
        self.assertEqual(self.visual_mock.await_count, 1)
        with patch.object(dp, 'ORIENTATION', 'square'):
            changed = await self.run_chunk()
        self.assertEqual(changed.audio_path, first.audio_path)
        self.assertEqual(self.audio_mock.await_count, 1)
        self.assertEqual(self.visual_mock.await_count, 2)

    async def test_voice_and_text_changes_preserve_previous_versions(self):
        first = await self.run_chunk()
        with patch.object(dp, 'TTS_VOICE_FILE', 'other.wav'):
            second = await self.run_chunk()
        third = await self.run_chunk('changed text')
        self.assertEqual(self.audio_mock.await_count, 3)
        self.assertEqual(len({first.audio_path, second.audio_path, third.audio_path}), 3)
        self.assertTrue(first.audio_path.exists())

    async def test_failed_visual_does_not_block_audio_and_is_retried(self):
        self.visual_mock.return_value = (None, None)
        first = await self.run_chunk()
        self.assertIsNotNone(first.audio_path)
        self.visual_mock.return_value = ('city', None)
        second = await self.run_chunk()
        self.assertEqual(second.search_prompt, 'city')
        self.assertEqual(self.audio_mock.await_count, 1)
        self.assertEqual(self.visual_mock.await_count, 2)

    async def test_corrupt_audio_regenerated(self):
        first = await self.run_chunk()
        first.audio_path.write_bytes(b'corrupted')
        await self.run_chunk()
        self.assertEqual(self.audio_mock.await_count, 2)

    async def test_failed_audio_is_retried(self):
        self.audio_mock.side_effect = None
        self.audio_mock.return_value = dp.ChunkResult(1, 'hello')
        await self.run_chunk()
        self.audio_mock.side_effect = self.make_audio
        result = await self.run_chunk()
        self.assertIsNotNone(result.audio_path)
        self.assertEqual(self.audio_mock.await_count, 2)

    async def test_media_cache_and_corruption(self):
        audio = await self.run_chunk()
        task = dp.build_media_tasks([audio], self.root)[0]
        self.assertEqual(task.audio_path, audio.audio_path)
        async def render(task, *args):
            task.video_path.write_bytes(b'video')
            return True
        with patch.object(dp, '_process_video_task_uncached', AsyncMock(side_effect=render)) as render_mock:
            args = (task, 1, None, '', '', None, asyncio.Semaphore(4))
            self.assertTrue(await dp.process_video_task(*args))
            self.assertTrue(await dp.process_video_task(*args))
            self.assertEqual(render_mock.await_count, 1)
            Path(dp.RUN_MEDIA[1]).write_bytes(b'bad')
            self.assertTrue(await dp.process_video_task(*args))
            self.assertEqual(render_mock.await_count, 2)

    async def test_partial_run_is_not_success(self):
        script = self.root / 'script.txt'
        script.write_text('one\n\ntwo')
        result = dp.ChunkResult(1, 'one', audio_path=self.root / 'one.wav', audio_duration=1.0)
        result.audio_path.write_bytes(b'audio')
        with patch.object(dp, 'SCRIPT_FILE', str(script)), \
             patch.object(dp, 'MEDIA_DOWNLOAD_ENABLED', False), \
             patch.object(dp, 'load_settings'), \
             patch.dict('os.environ', {'GEMINI_API_KEY': 'test-placeholder'}), \
             patch.object(dp.genai, 'Client'), \
             patch.object(dp, 'check_ffmpeg_tools', AsyncMock()), \
             patch.object(dp, 'process_all_chunks', AsyncMock(return_value=[result, dp.ChunkResult(2, 'two')])):
            code = await dp.async_main()
        self.assertEqual(code, 1)
        summary = json.loads((dp.PROJECT.path / 'run_summary.json').read_text())
        self.assertEqual(summary['status'], 'partial')
        self.assertEqual(summary['expected'], 2)
        self.assertEqual(summary['completed'], 1)

    async def test_tts_requests_are_serialized(self):
        active = 0
        maximum = 0
        def generate(*args):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            time.sleep(0.02)
            active -= 1
        with patch.object(dp, 'generate_audio_with_chatterbox', generate):
            await asyncio.gather(*(dp.generate_audio_with_chatterbox_async('x', self.root / 'x') for _ in range(4)))
        self.assertEqual(maximum, 1)


if __name__ == '__main__':
    unittest.main()
