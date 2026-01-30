"""
Logging and performance monitoring utilities
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import numpy as np

from .visualization import Color

if TYPE_CHECKING:
    from types import TracebackType


class ColoredFormatter(logging.Formatter):
    """
    Custom `logging.Formatter` coloring logs based on the severity level
    """

    FORMAT: ClassVar[str] = '%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)-20s | %(message)s'
    FORMATS: ClassVar[dict[int, str]] = {
        logging.DEBUG: Color.GREY.ansi(FORMAT),
        logging.INFO: Color.INFO.ansi(FORMAT),
        logging.WARNING: Color.WARNING.ansi(FORMAT),
        logging.ERROR: Color.ERROR.ansi(FORMAT),
        logging.CRITICAL: Color.ERROR.ansi(FORMAT)
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')

        return formatter.format(record)


def get_logger(name: str = 'System', level: int = logging.INFO) -> logging.Logger:
    """
    Generate a `logging.Logger` instance with `ColoredFormatter` as default

    :param name: Name of the logger
    :param level: Baseline severity level; logs with lower level will not be displayed
    :return: A `logging.Logger` instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(ColoredFormatter())
        logger.addHandler(ch)

    return logger


class PerformanceMonitor:
    """
    Performance monitor recording operation durations, preserving value history
    and calculating moving metrics
    """

    def __init__(self, history_size: int | None = 1000, report_interval: float = 5.0) -> None:
        """
        Initialize a performance monitor

        :param history_size: Number of values kept in history for each metric;
                             if `None`, the history will be unlimited
        :param report_interval: Number of seconds between periodic reports
        """

        if history_size is not None and history_size <= 0:
            raise ValueError

        # Preserve thread safety
        self._lock = threading.Lock()

        self.history_size: int | None = history_size
        self._metrics: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=history_size))
        self._n: dict[str, int] = defaultdict(int)

        self.report_interval: float = report_interval
        self._last_report: float = 0.0

    def record(self, name: str, duration: float) -> None:
        """
        Record an operation duration

        :param name: Name of the operation
        :param duration: Time in seconds
        """

        with self._lock:
            self._metrics[name].append(duration)
            self._n[name] += 1

    def get_summary(self) -> dict[str, dict[str, float]]:
        """
        Compute a summary for each recorded metric

        :return: Summary in the form of dictionary with metric names as keys
        """

        summary: dict[str, dict[str, float]] = {}

        with self._lock:
            for name, durations in self._metrics.items():
                if not durations:
                    continue

                n = self._n[name]
                avg_time = np.mean(durations)
                std_time = np.std(durations)
                min_time = min(durations)
                max_time = max(durations)
                fps = 1.0 / avg_time if avg_time > 0 else 0.0

                summary[name] = {
                    'n': n,
                    'avg_ms': avg_time * 1000,
                    'std_ms': std_time * 1000,
                    'min_ms': min_time * 1000,
                    'max_ms': max_time * 1000,
                    'fps': fps
                }

        return summary

    def log_periodically(self, logger: logging.Logger) -> None:
        """
        Log the performance summary if in reporting interval

        :param logger: Logger to record to
        """

        now = time.time()
        if now - self._last_report < self.report_interval:
            return

        summary = self.get_summary()
        self._last_report = now

        report = [
            f'--- Performance Report (Last {self.report_interval}s) ---',
            f'{"Operation":<70} | {"FPS":<12} | {"Avg [ms]":<20} | {"Max [ms]":<12} | {"Min [ms]":<12} | {"N":<12}',
            '-' * 160
        ]

        for name in sorted(summary.keys()):
            data: dict[str, float] = summary[name]
            avg_ms = f'{data["avg_ms"]:.2f} +- {data["std_ms"]:.2f}'
            min_ms = data['min_ms']
            max_ms = data['max_ms']
            fps = data['fps']
            n = data['n']

            report.append(f'{name:<70} | {fps:<12.2f} | {avg_ms:<20} | {max_ms:<12.2f} | {min_ms:<12.2f} | {n:<12,d}')

        report.append('-' * 160)
        logger.info('\n'.join(report))


class timer:
    """
    Context manager and decorator measuring elapsed time
    of the operations contained within it
    """

    def __init__(
            self,
            name: str = '',
            logger: logging.Logger | None = None,
            monitor: PerformanceMonitor | None = None,
            level: int = logging.DEBUG,
            logger_attr: str = 'logger',
            monitor_attr: str = 'monitor'
    ) -> None:
        """
        Initialize the timer either as a context manager or as a decorator of a function or a method.
        If used as a method decorator, the `timer` will look for a logger and a performance monitor
        as instance attributes named according to `logger_attr` and `monitor_attr`, respectively.

        :param name: Name of the timer or operation
        :param logger: Optional logger to record the measured values to
        :param monitor: Optional `PerformanceMonitor` to record the measured values with
        :param level: Baseline logging severity level
        :param logger_attr: Name of the instance attribute containing a `logging.Logger` instance
                            to use if the timer is used as a method decorator
        :param monitor_attr: Name of the instance attribute containing a `PerformanceMonitor` instance
                             to use if the timer is used as a method decorator
        """

        self.name: str = name

        self.logger: logging.Logger = logger
        self.level: int = level
        self.monitor: PerformanceMonitor = monitor

        self.logger_attr: str = logger_attr
        self.monitor_attr: str = monitor_attr

    def __enter__(self) -> timer:
        self.start: float = time.perf_counter()
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None
    ) -> None:
        self.time = time.perf_counter() - self.start

        if self.monitor is not None:
            self.monitor.record(self.name, self.time)

        msg = f'Operation{" " + self.name if self.name else ""}: {self.time * 1000:.2f} ms'

        if self.logger is not None:
            self.logger.log(
                level=self.level,
                msg=msg
            )

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Generate a `timer` instance for a decorated function
        """

        @functools.wraps(func)
        def wrapper(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
            report_name = self.name
            if not report_name:
                f_name = func.__name__
                c_name = (
                    args[0].__class__.__name__
                    if args and hasattr(args[0], '__class__')
                    else ''
                )
                report_name = f'{c_name}.{f_name}'

            logger = self.logger
            monitor = self.monitor

            if args:
                instance: object = args[0]

                if logger is None and hasattr(instance, self.logger_attr):
                    logger = getattr(instance, self.logger_attr)

                if monitor is None and hasattr(instance, self.monitor_attr):
                    monitor = getattr(instance, self.monitor_attr)

            t = timer(
                name=report_name,
                logger=logger,
                monitor=monitor,
                level=self.level
            )

            with t:
                return func(*args, **kwargs)

        return wrapper
