import datetime
import functools
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from typing import List, Optional

import torch
import torch.distributed as dist
import wandb

from .distributed import get_global_rank, is_enabled, is_main_process

logger = logging.getLogger("STORM")


class SmoothedValue:
    """Tracks a series of values and computes smoothed statistics (e.g., median, average)."""

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, num=1):
        """Updates the tracked values with a new value."""
        self.deque.append(value)
        self.total += value * num
        self.count += num

    def synchronize_between_processes(self):
        """Synchronizes the metric values across distributed processes."""
        if not is_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        self.count, self.total = int(t[0].item()), t[1].item()

    @property
    def median(self):
        """Computes the median of the tracked values."""
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        """Computes the average of the tracked values."""
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        """Computes the global average across all processes."""
        return self.total / self.count

    @property
    def max(self):
        """Returns the maximum tracked value."""
        return max(self.deque)

    @property
    def value(self):
        """Returns the most recent value."""
        return self.deque[-1]

    def __str__(self):
        """Formats the smoothed statistics as a string."""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter="\t", output_file=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.median for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time, data_time = SmoothedValue(fmt="{avg:.4f}"), SmoothedValue(fmt="{avg:.4f}")

        n_iterations = n_iterations or len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "elapsed: {elapsed_time_str}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                self.dump_in_output_file(
                    iteration=i, iter_time=iter_time.avg, data_time=data_time.avg
                )
                eta_seconds = iter_time.global_avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                elapsed_time = time.time() - start_time
                elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))

                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            elapsed_time_str=elapsed_time_str,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / n_iterations
            )
        )


class WandbLogger:
    def __init__(self, args, resume="must", id=None):
        if id is None:
            resume = "allow" if resume else "never"
        wandb.init(
            config=args,
            entity=args.entity,
            project=args.project,
            name=args.exp_name,
            dir=args.log_dir,
            resume=resume,
            id=id,
        )
        self.run_id = wandb.run.id
        self.step = 0
        wandb.run.save()

    @staticmethod
    def wandb_safe_log(*args, **kwargs):
        """Safely logs metrics to Wandb, handling errors."""
        try:
            wandb.log(*args, **kwargs)
        except (wandb.CommError, BrokenPipeError):
            logger.error("Wandb logging failed, skipping...")

    def set_step(self, step=None):
        """Sets the logging step."""
        self.step = step if step is not None else self.step + 1

    def update(self, metrics):
        """Updates metrics in Weights & Biases."""
        log_dict = {
            k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in metrics.items()
            if v is not None
        }
        self.wandb_safe_log(log_dict, step=self.step)

    def flush(self):
        pass

    def finish(self):
        try:
            wandb.finish()
        except (wandb.CommError, BrokenPipeError):
            logger.error("Wandb failed to finish")


@functools.lru_cache()
def _configure_logger(
    name: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
    output: Optional[str] = None,
    time_string: Optional[str] = None,
    rank0_log: bool = True,
) -> logging.Logger:
    """
    Configure a logger with optional file and console outputs.

    Args:
        name: Name of the logger to configure. Defaults to the root logger.
        level: Logging level (e.g., DEBUG, INFO). Default is DEBUG.
        output: Path to save logs. If None, logs are not saved.
            - If it ends with ".txt" or ".log", treated as a file name.
            - Otherwise, logs are saved to `output/logs/log.txt`.
        time_string: Timestamp string to append to log filenames.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Loosely match Google glog format:
    #   [IWEF]yyyymmdd hh:mm:ss.uuuuuu threadid file:line] msg
    # but use a shorter timestamp and include the logger name:
    #   [IWEF]yyyymmdd hh:mm:ss logger threadid file:line] msg
    fmt_prefix = "%(levelname).1s%(asctime)s %(name)s %(filename)s:%(lineno)s] "
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # stdout logging for main worker only
    if is_main_process():
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # File logging
    # Only create a file handler if:
    #   - There is an output path, AND
    #   - (We are rank 0 or rank0_log=False)
    if output and (is_main_process() or not rank0_log):
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            if time_string is None:
                filename = os.path.join(output, "logs", "log.txt")
            else:
                filename = os.path.join(output, "logs", f"log_{time_string}.txt")

        # If it's not rank 0 but rank0_log=False, append the rank ID
        if not is_main_process() and not rank0_log:
            global_rank = get_global_rank()
            filename = f"{filename}.rank{global_rank}"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_logging(
    output: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    capture_warnings: bool = True,
    time_string: Optional[str] = None,
) -> None:
    """
    Setup logging with optional console and file handlers.

    Args:
        output: Path to save log files. If None, logs are not saved.
            - If it ends with ".txt" or ".log", treated as a file name.
            - Otherwise, logs are saved to `output/logs/log.txt`.
        name: Name of the logger to configure. Defaults to the root logger.
        level: Logging level (e.g., DEBUG, INFO). Default is DEBUG.
        capture_warnings: Whether Python warnings should be captured as logs.
        time_string: Timestamp string to append to log filenames.
    """
    logging.captureWarnings(capture_warnings)
    _configure_logger(name, level=level, output=output, time_string=time_string)
