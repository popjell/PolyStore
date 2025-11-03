"""Tests for backend registry lazy instantiation and cleanup behavior."""
from polystore import backend_registry as br


def test_get_backend_instance_and_cleanup():
    # instantiate memory backend and verify caching
    mem1 = br.get_backend_instance("memory")
    mem2 = br.get_backend_instance("memory")
    assert mem1 is mem2

    # instantiate disk to ensure it exists
    d1 = br.get_backend_instance("disk")

    # cleanup all backends should clear the cache
    br.cleanup_all_backends()

    # after cleanup the instances should be new objects
    mem3 = br.get_backend_instance("memory")
    assert mem3 is not mem1
