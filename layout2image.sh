#!/bin/bash
#first run Android/Sdk/emulator/./emulator -avd Nexus_5X_API_29 
adb shell monkey -p com.example.randomizer -v  1
echo "Randomizer started in emulator"
Android/Sdk/platform-tools/./adb -s emulator-5554 exec-out screencap -p > screen.png
echo "screencap done"
declare -i count=0
for file in AndroidStudioProjects/Randomizer/app/src/main/res/layout/screens/*
do
	echo $file
	#sed '24d' file
	#line=$(sed -n '/#0ffffff/=' $file)
        #echo "**************"
	#echo $line	
	sed -i '/#0ffffff/d' $file
	base_name=${file##*/}
	file_name=${base_name%.*}
	echo $file_name
	mv $file AndroidStudioProjects/Randomizer/app/src/main/res/layout/activity_main.xml
	echo "finish rename"
	cd AndroidStudioProjects/Randomizer/
	./gradlew installDebug
	cd ../..
	adb shell monkey -p com.example.randomizer -v 1
	echo "Randomizer restarted in emulator"
	echo "Installed debug"
	#count=$((count+1))
	name="screens/${file_name}.png"
	echo $name
	Android/Sdk/platform-tools/adb -s emulator-5554 exec-out screencap -p > $name 
	echo "second screencap"
	$count = $count + 1

done
#mv AndroidStudioProjects/Randomizer/app/src/main/res/layout/activity_main.xml AndroidStudioProjects/Randomizer/app/src/main/res/layout/activity_main1_aux.xml
#echo "first rename"
#mv AndroidStudioProjects/Randomizer/app/src/main/res/layout/activity_main1.xml AndroidStudioProjects/Randomizer/app/src/main/res/layout/activity_main.xml
#echo "second rename"
#mv AndroidStudioProjects/Randomizer/app/src/main/res/layout/activity_main1_aux.xml AndroidStudioProjects/Randomizer/app/src/main/res/layout/activity_main1.xml
#echo "finish rename"
#cd AndroidStudioProjects/Randomizer/
#./gradlew installDebug
#cd ../..
#adb shell monkey -p com.example.randomizer -v 1
#echo "Randomizer restarted in emulator"
#echo "Installed debug"
#Android/Sdk/platform-tools/adb -s emulator-5556 exec-out screencap -p > screen1.png
#echo "second screencap"
