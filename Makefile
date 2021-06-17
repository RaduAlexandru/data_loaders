all:
	echo "Building DataLoaders"
	python3 -m pip install -v --user --editable ./

clean:
	python3 -m pip uninstall dataloaders
	# catkin clean dataloaders
	rm -rf build *.egg-info build dataloaders*.so libdataloaders_cpp.so
