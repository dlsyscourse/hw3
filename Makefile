.PHONY: lib pybind clean format all

all: lib

# build/ remembers the machine that configured it (CUDA paths, GPU arch, -march=native objects).
# On Colab it lives in Drive and outlives the runtime, so start over when the machine changes.
HOST := $(shell hostname)

lib:
	@if [ -d build ] && [ "$$(cat build/.host 2>/dev/null)" != "$(HOST)" ]; then \
		echo "build/ was configured on a different machine; rebuilding from scratch"; \
		rm -rf build python/needle/backend_ndarray/ndarray_backend*.so; \
	fi
	@mkdir -p build
	@echo "$(HOST)" > build/.host
	@cd build; cmake ..
	@cd build; $(MAKE) --no-print-directory

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so
