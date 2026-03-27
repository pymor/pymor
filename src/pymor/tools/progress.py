# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from time import perf_counter

from rich.console import Group
from rich.live import Live
from rich.progress import Progress
from rich.text import Text


class ProgressDisplay:

    def add_task(self, title=None, total=None):
        return Task(self, 0)

    def track(self, iterable, title=None, total=None):
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                pass
        with self.add_task(title=title, total=total) as task:
            for i, v in enumerate(iterable):
                yield v
                task.update(i+1)

    def _update_task(self, task_id, value, total=None):
        pass

    def _task_finished(self, task_id):
        pass


class Task:
    def __init__(self, progress_display, id):
        self.progress_display, self.id = progress_display, id
        self.finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()

    def __del__(self):
        self.finish()

    def update(self, value, total=None):
        self.progress_display._update_task(self.id, value, total=total)

    def finish(self):
        if not self.finished:
            self.progress_display._task_finished(self.id)
            self.finished = True


class RichProgressDisplay(ProgressDisplay):

    def __init__(self):
        self.max_tasks = 0
        self.progress = Progress(auto_refresh=False)
        self.tasks = []
        self.finished_tasks = {}
        self.last_update_max_tasks = perf_counter()
        self.longest_title = 0
        self.live = Live(get_renderable=self._get_rederable, transient=True)
        self.started = False

    def add_task(self, title=None, total=None):
        for task_id in self.finished_tasks:
            self.progress.remove_task(task_id)
            self.tasks.remove(task_id)
        self.finished_tasks.clear()
        if title is not None:
            self.longest_title = max(self.longest_title, len(title))
        for task_id, task in zip(self.progress.task_ids, self.progress.tasks, strict=True):
            self.progress.update(task_id, description=f'{task.description:{self.longest_title}}')
        task_id = self.progress.add_task(f'{title:{self.longest_title}}', total=total)
        self.tasks.append(task_id)
        self.max_tasks = max(self.max_tasks, len(self.tasks))
        if not self.started:
            self.live.start()
        return Task(self, task_id)

    def _update_task(self, task_id, value, total=None):
        self.progress.update(task_id, completed=value, total=total)

    def _get_rederable(self):
        tic = perf_counter()
        to_remove = []
        for tid, t_finished in self.finished_tasks.items():
            if tic - t_finished > 2:
                to_remove.append(tid)
        for tid in to_remove:
            self.progress.remove_task(tid)
            self.tasks.remove(tid)
            del self.finished_tasks[tid]
        if tic - self.last_update_max_tasks > 2:
            self.max_tasks = len(self.tasks)
            self.last_update_max_tasks = tic
        if self.max_tasks > len(self.tasks):
            return Group(self.progress.get_renderable(), Text('\n' * (self.max_tasks - len(self.tasks) - 1)))
        else:
            return self.progress.get_renderable()

    def _task_finished(self, task_id):
        for tid, task in zip(self.progress.task_ids, self.progress.tasks, strict=True):
            if tid == task_id:
                if task.completed == 0:
                    if not task.total:
                        self.progress.update(task_id, completed=1, total=1)
                    else:
                        self.progress.update(task_id, completed=task.total)
                else:
                    self.progress.update(task_id, total=task.completed)
                break
        self.finished_tasks[task_id] = perf_counter()
        if len(self.tasks) == len(self.finished_tasks):
            try:
                self.live.stop()
                self.started = False
            except AttributeError:
                pass


progress_display = RichProgressDisplay()


def get_progress_display():
    return progress_display
