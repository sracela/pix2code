#!/bin/bash

# First run in terminal: ~/Android/Sdk/emulator/./emulator -avd Pixel_2_API_29
~/Android/Sdk/platform-tools/./adb shell monkey -p com.example.randomizer -v  1
echo "Randomizer started in emulator"
~/Android/Sdk/platform-tools/./adb -s emulator-5554 exec-out screencap -p > screen.png
echo "screencap done"

for file in android/Randomizer/app/src/main/res/layout/screens/*
do
	echo $file
	sed -i '/#0ffffff/d' $file
	base_name=${file##*/}
	file_name=${base_name%.*}
	echo $file_name
	mv $file android/Randomizer/app/src/main/res/layout/activity_main.xml
	echo "finish rename"
	cd android/Randomizer/
	./gradlew installDebug
	cd ../..
	~/Android/Sdk/platform-tools/./adb shell monkey -p com.example.randomizer -v 1
	echo "Randomizer restarted in emulator"
	echo "Installed debug"
	name="code/original/screens/${file_name}.png"
	echo $name
	sleep 4
	~/Android/Sdk/platform-tools/./adb -s emulator-5554 exec-out screencap -p > $name 
	echo "second screencap"

done

