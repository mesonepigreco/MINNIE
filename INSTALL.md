# INSTALL GUIDE

## Libraries needed
To compile (and install) ANNIE you need to have the developer version of the following libraries

1. GLS (GNU Scientific Library)
2. Libconfig++
3. BOOST (system and filesystem extension)
4. (Optional) Python-dev

In most unix system, these libraries can be installed by a simple command (e.g. apt-get install under ubuntu). 
Look for the libraries in the package-manager of your distribution with the '-dev' suffix.

## Compiling

If you correctly installed all the libraries and your system is "standard", then all you need to do is:
```bash
make 
```

The software will be compiled and placed into the bin folder. If something goes wrong, and you are shure to have
correctly installed everything, you can try to edit the make.inc file.

In this file all the libraries and compiler options are specified. You can try to manually set the linking
path of your libraries in the directory in which you have compiled them. This is needed when you try to install
ANNIE on a system where you do not have root privileges, therefore you must compile the needed libraries by yourself.

To make the program callable from outside the bin directory, you can add an alias to your .bashrc:

```bash
echo "alias ANNIE=`pwd`/bin/ANNIE.exe" >> $HOME/.bashrc
source $HOME/.bashrc
```
This command must be executed inside the ANNIE main directory.

NOTE: The software is still not complete. It will not generate the ANNIE executable when compiled, but just some
debugging stuff.
