"""Queue one Bash training script behind an active Linux managed-local DP7 run."""

import argparse
import fcntl
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import sys
import time


def supervisor_args(args, run):
    # Match the CLI entry point, not a shell that happens to quote its command.
    for i, token in enumerate(args[:-1]):
        if Path(token).name in ("specforge", "specforge.cli") and args[i + 1] == "train":
            break
    else:
        return False
    if "--role" in args or "--plan" in args:
        return False
    overrides = dict(a.split("=", 1) for a in args[i + 2:] if "=" in a)
    for key, expected in (("output_dir", run), ("deployment.disaggregated.control_dir", run / "control")):
        path = Path(overrides.get(key, ""))
        if not path.is_absolute() or path.resolve() != expected:
            return False
    return True


def read_args(proc):
    return [os.fsdecode(a) for a in (proc / "cmdline").read_bytes().split(b"\0") if a]


def attach(run):
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != os.getuid() or not supervisor_args(read_args(proc), run):
                continue
            matches.append(proc)
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"匹配到 {len(matches)} 个正在运行的主进程，请核对本轮 output_dir：{run}")
    proc = matches[0]
    fd = os.pidfd_open(int(proc.name))
    return proc, fd


def check_failures(control):
    for suffix in ("failed", "consumer_failed"):
        path = control / f"refs.jsonl.{suffix}"
        if path.exists():
            raise RuntimeError(f"{path.name}: {path.read_text().strip()}")


def final_checkpoint(run, run_id):
    link = run / f"{run_id}-latest"
    path = link.resolve(strict=True)
    if (path.parent != run or not re.fullmatch(re.escape(run_id) + r"-step\d+", path.name)
            or not (path / "training_state.pt").is_file()):
        raise RuntimeError(f"找不到已提交的最终训练 checkpoint：{link}")
    return path


def consumer_running(proc):
    # torchrun is a direct child of the single-node managed supervisor.
    try:
        children = (proc / "task" / proc.name / "children").read_text().split()
    except FileNotFoundError:
        return False
    for pid in children:
        try:
            args = read_args(Path("/proc") / pid)
        except FileNotFoundError:
            continue
        if any(pair == ("--role", "consumer") for pair in zip(args, args[1:])):
            return True
    return False


def busy_resources():
    # This recipe uses GPU 0 for capture and GPUs 1-7 for the trainer.
    query = subprocess.check_output(
        ["nvidia-smi", "-i", "0,1,2,3,4,5,6,7", "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], text=True,
    ).strip()
    if query:
        return f"GPU 计算进程 {', '.join(query.splitlines())}"
    for port in (30000, 35551, 35880, 35903):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                return f"端口 {port} ({exc})"
    return None


def watch(args):
    run = args.run.resolve(strict=True)
    control = run / "control"
    script = args.next_script.resolve(strict=True)
    subprocess.run(["bash", "-n", str(script)], check=True)
    with (control / ".next-training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("这轮已有接力脚本在运行") from None
        marker = control / ".next-training-started"
        if marker.exists():
            raise RuntimeError(f"这轮已提交过下一任务：{marker.read_text().strip()}")
        check_failures(control)
        proc, fd = attach(run)
        try:
            args_now = read_args(proc)
            if not supervisor_args(args_now, run):
                raise RuntimeError("匹配到的主进程已退出")
            run_id = next((a.partition("=")[2] for a in args_now if a.startswith("run_id=")), run.name)
            poll = select.poll()
            poll.register(fd, select.POLLIN)
            print(f"已监控 {run.name} (PID {proc.name})，下一轮 {script}", flush=True)
            cleanup_since = None
            sent = False
            while not poll.poll(0):
                check_failures(control)
                if ((control / "refs.jsonl.consumer_done").exists()
                        and (control / "refs.jsonl.closed").exists()
                        and not consumer_running(proc)):
                    if cleanup_since is None:
                        print(f"训练完成，已保存 {final_checkpoint(run, run_id).name}，等待服务退出", flush=True)
                        cleanup_since = time.monotonic()
                    if (args.stop_cleanup_after is not None and not sent
                            and time.monotonic() - cleanup_since >= args.stop_cleanup_after):
                        check_failures(control)
                        print("收尾超时，向本轮 SpecForge 主进程发送 SIGTERM", flush=True)
                        try:
                            signal.pidfd_send_signal(fd, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        sent = True
                poll.poll(10000)
        finally:
            os.close(fd)
        check_failures(control)
        if not all((control / f"refs.jsonl.{s}").exists() for s in ("consumer_done", "closed")):
            raise RuntimeError("主进程提前退出，缺少训练完成或采集结束标记")
        checkpoint = final_checkpoint(run, run_id)
        last_busy = None
        while busy := busy_resources():
            if busy != last_busy:
                print(f"等待资源释放：{busy}", flush=True)
                last_busy = busy
            time.sleep(10)
        check_failures(control)
        with marker.open("x") as stream:
            stream.write(f"{script}\n")
        print(f"{checkpoint.name} 已保存，GPU 和端口已释放，启动 {script}", flush=True)
        return subprocess.call(["bash", str(script)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="当前训练的完整 output_dir")
    parser.add_argument("next_script", type=Path, help="下一轮 Bash 启动脚本")
    parser.add_argument("--stop-cleanup-after", type=int, metavar="SECONDS",
                        help="训练成功且 trainer 退出后，给旧服务的收尾时间；超时通知主进程关闭服务")
    args = parser.parse_args()
    if args.stop_cleanup_after is not None and args.stop_cleanup_after <= 0:
        parser.error("--stop-cleanup-after 必须大于 0")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        parser.error("需要在训练服务器上用 Linux Python 3.9+ 运行")
    try:
        return watch(args)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"接力停止：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
