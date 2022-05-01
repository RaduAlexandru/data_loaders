all:
	echo "Building DataLoaders"
	python3 -m pip install -v --user --editable ./

clean:
	python3 -m pip uninstall dataloaders
	-catkin clean data_loaders #this command is ok to fail for example when caktin is not present
	rm -rf build *.egg-info build dataloaders*.so libdataloaders_cpp.so
