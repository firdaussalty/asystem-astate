DEPS_ARGS ?=

deps:
	sh scripts/install_deps.sh $(DEPS_ARGS)

release:
	sh build.sh release

install: release
	sh scripts/install_as_pylib.sh

develop:
	sh build.sh develop
	sh scripts/install_as_pylib.sh

test: proto develop
	sh build.sh test

clean:
	rm -rf build

.PHONY: deps release install develop test clean
