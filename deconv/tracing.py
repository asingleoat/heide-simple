"""
Simple tracing utility for performance analysis.

Usage:
    from deconv.tracing import tracer, trace

    # Enable tracing
    tracer.enable()

    # Use as context manager
    with trace("operation_name"):
        do_something()

    # Or use decorator
    @trace("function_name")
    def my_function():
        pass

    # Print summary
    tracer.print_summary()
"""

import time
from contextlib import contextmanager
from functools import wraps
from collections import defaultdict


class Tracer:
    """Global tracer for performance analysis."""

    def __init__(self):
        self._enabled = False
        self._spans = defaultdict(lambda: {"count": 0, "total_time": 0.0})
        self._stack = []  # Stack of (name, start_time) for nested spans
        self._call_order = []  # Track order spans were first seen

    def enable(self):
        """Enable tracing."""
        self._enabled = True
        self._spans.clear()
        self._stack.clear()
        self._call_order.clear()

    def disable(self):
        """Disable tracing."""
        self._enabled = False

    def is_enabled(self):
        """Check if tracing is enabled."""
        return self._enabled

    def reset(self):
        """Reset all accumulated data."""
        self._spans.clear()
        self._stack.clear()
        self._call_order.clear()

    def enter(self, name):
        """Enter a traced span."""
        if not self._enabled:
            return
        self._stack.append((name, time.perf_counter()))

    def exit(self, name):
        """Exit a traced span."""
        if not self._enabled:
            return
        if not self._stack:
            return

        end_time = time.perf_counter()
        span_name, start_time = self._stack.pop()

        # Build full hierarchical name
        if self._stack:
            parent_names = [s[0] for s in self._stack]
            full_name = "/".join(parent_names + [span_name])
        else:
            full_name = span_name

        elapsed = end_time - start_time

        if full_name not in self._spans:
            self._call_order.append(full_name)

        self._spans[full_name]["count"] += 1
        self._spans[full_name]["total_time"] += elapsed

    def record(self, name, elapsed):
        """Record a timing directly (for manual instrumentation)."""
        if not self._enabled:
            return

        # Build full hierarchical name
        if self._stack:
            parent_names = [s[0] for s in self._stack]
            full_name = "/".join(parent_names + [name])
        else:
            full_name = name

        if full_name not in self._spans:
            self._call_order.append(full_name)

        self._spans[full_name]["count"] += 1
        self._spans[full_name]["total_time"] += elapsed

    def get_summary(self):
        """Get timing summary as a list of dicts."""
        if not self._spans:
            return []

        results = []
        for name in self._call_order:
            data = self._spans[name]
            results.append({
                "name": name,
                "count": data["count"],
                "total_time": data["total_time"],
                "avg_time": data["total_time"] / data["count"] if data["count"] > 0 else 0,
            })

        return results

    def print_summary(self):
        """Print a formatted timing summary."""
        summary = self.get_summary()
        if not summary:
            print("No tracing data collected.")
            return

        # Calculate total time from top-level spans
        top_level_time = sum(
            s["total_time"] for s in summary if "/" not in s["name"]
        )

        print("\n" + "=" * 80)
        print("TIMING SUMMARY")
        print("=" * 80)
        print(f"{'Span':<50} {'Count':>8} {'Total (s)':>12} {'Avg (ms)':>12} {'%':>6}")
        print("-" * 80)

        for span in summary:
            name = span["name"]
            # Indent based on hierarchy depth
            depth = name.count("/")
            display_name = "  " * depth + name.split("/")[-1]
            if len(display_name) > 50:
                display_name = display_name[:47] + "..."

            pct = (span["total_time"] / top_level_time * 100) if top_level_time > 0 else 0

            print(f"{display_name:<50} {span['count']:>8} {span['total_time']:>12.4f} "
                  f"{span['avg_time']*1000:>12.3f} {pct:>5.1f}%")

        print("-" * 80)
        print(f"{'Total traced time:':<50} {'':<8} {top_level_time:>12.4f}")
        print("=" * 80 + "\n")


# Global tracer instance
tracer = Tracer()


@contextmanager
def trace(name):
    """Context manager for tracing a code block."""
    tracer.enter(name)
    try:
        yield
    finally:
        tracer.exit(name)


def trace_function(name=None):
    """Decorator for tracing a function."""
    def decorator(func):
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer.enter(span_name)
            try:
                return func(*args, **kwargs)
            finally:
                tracer.exit(span_name)

        return wrapper
    return decorator
