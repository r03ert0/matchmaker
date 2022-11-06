# Match maker

RT, September 2022

Match 2 meshes using the `matchmesh` algorithm without providing a starting point, but trying instead all 24 axis-aligned rotation matrices to find the best one.

## Installation

1. After cloning the repository `git submodule update --init --recursive`, to get all the submodules.

2. Compile GraphiteThree

First, in the file `GraphiteThree/geogram/CMakeOptions.txt` comment out the last line to be like this:
```
# set(GEOGRAM_LIB_ONLY ON)
```

Then:

```
cd GraphiteThree
bash configure.sh
cd build/Darwin-clang-dynamic-Release
make -j8
```

3. Compile meshgeometry

4. Compule meshparam