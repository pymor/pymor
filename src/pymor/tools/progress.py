# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import os
from abc import ABC, abstractmethod
from time import perf_counter

from rich.console import Group
from rich.live import Live
from rich.progress import Progress
from rich.text import Text


class ProgressDisplay(ABC):
    """Displays multiple progress bars."""

    @abstractmethod
    def add_task(self, label=None, total=None):
        """Add new progress bar for given task.

        Parameters
        ----------
        label
            If not `None`, the label to use for the progress bar for this task.
        total
            If not `None`, the total units of work to be performed by this task.

        Returns
        -------
        task
            The :class:`Task` object representing the progress bar.
        """
        pass

    def track(self, iterable, label=None, total=None):
        """Track iteration over an iterable.

        Consecutively yields the items of the iterable and updates the
        corresponding progress bar.

        Parameters
        ----------
        iterable
            The iterable to track.
        label
            If not `None`, the label for the corresponding progress bar.
        total
            If not `None`, the the length of the iterable.
        """
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                pass
        with self.add_task(label=label, total=total) as task:
            for i, v in enumerate(iterable):
                yield v
                task.update(i+1)

    @abstractmethod
    def _update_task(self, task_id, value, total=None):
        pass

    @abstractmethod
    def _task_finished(self, task_id):
        pass


class Task:
    """A task corresponding to a progress bar.

    Created by a :class:`ProgressDisplay`.

    Can be used as a context manager. In that case, the task is
    automatically :meth:`finished <Task.finish>` when the context
    is left.
    """

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
        """Update progress of the task.

        Parameters
        ----------
        value
            The units of work that have been finished.
        total
            If not `None` a (new) value for the total number of unit of
            work to be performed.
        """
        self.progress_display._update_task(self.id, value, total=total)

    def finish(self):
        """Mark task as finished."""
        if not self.finished:
            self.progress_display._task_finished(self.id)
            self.finished = True


class DummyProgressDisplay(ProgressDisplay):

    def add_task(self, label=None, total=None):
        return Task(self, 0)

    def _update_task(self, task_id, value, total=None):
        pass

    def _task_finished(self, task_id):
        pass


class RichProgressDisplay(ProgressDisplay):

    TASK_LINGER_TIME = 2
    TASK_PANE_UPDATE_TIME = 2

    def __init__(self):
        self.max_tasks = 0
        self.progress = Progress(auto_refresh=False)
        self.tasks = []
        self.finished_tasks = {}
        self.last_update_max_tasks = perf_counter()
        self.longest_label = 0
        self.live = Live(get_renderable=self._get_rederable, transient=True)
        self.started = False

    def add_task(self, label=None, total=None):
        for task_id in self.finished_tasks:
            self.progress.remove_task(task_id)
            self.tasks.remove(task_id)
        self.finished_tasks.clear()
        if label is not None:
            self.longest_label = max(self.longest_label, len(label))
        for task_id, task in zip(self.progress.task_ids, self.progress.tasks, strict=True):
            self.progress.update(task_id, description=f'{task.description:{self.longest_label}}')
        task_id = self.progress.add_task(f'{label:{self.longest_label}}', total=total)
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
            if tic - t_finished > self.TASK_LINGER_TIME:
                to_remove.append(tid)
        for tid in to_remove:
            self.progress.remove_task(tid)
            self.tasks.remove(tid)
            del self.finished_tasks[tid]
        if tic - self.last_update_max_tasks > self.TASK_PANE_UPDATE_TIME:
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


progress_display = RichProgressDisplay() if int(os.environ.get('PYMOR_PROGRESS', 0)) == 1 else DummyProgressDisplay()


def get_progress_display():
    """Returns the currently active :class:`ProgressDisplay`."""
    return progress_display


def add_task(label=None, total=None):
    """Add new progress bar for given task to active :class:`ProgressDisplay`.

    Parameters
    ----------
    label
        If not `None`, the label to use for the progress bar for this task.
    total
        If not `None`, the total units of work to be performed by this task.

    Returns
    -------
    task
        The :class:`Task` object representing the progress bar.
    """
    return get_progress_display().add_task(label=label, total=total)


def track(iterable, label=None, total=None):
    """Track iteration over an iterable in the active :class:`ProgressDisplay`.

    Consecutively yields the items of the iterable and updates the
    corresponding progress bar.

    Parameters
    ----------
    iterable
        The iterable to track.
    label
        If not `None`, the label for the corresponding progress bar.
    total
        If not `None`, the the length of the iterable.
    """
    return get_progress_display().track(iterable, label=label, total=total)
