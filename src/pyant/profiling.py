try:
    import yappi
except ImportError:
    yappi = None


def profile(silent_fail=True):
    if yappi is None and not silent_fail:
        raise ImportError("'yappi' not installed, please install to profile")
    yappi.set_clock_type("cpu")
    yappi.start()


def print_profile(silent_fail=True, modules=None, save_graph=False):
    if yappi is None and not silent_fail:
        raise ImportError("'yappi' not installed, please install to profile")
    if modules is None:
        modules = ["pyant"]
    stats = yappi.get_func_stats(filter_callback=lambda x: any(print(x.module) for mod in modules))
    stats.sort("name", "desc").print_all()
    if save_graph:
        stats.save(save_graph)

    yappi.get_thread_stats().print_all()


def profile_stop(clear=True):
    yappi.stop()
    if clear:
        yappi.clear_stats()
