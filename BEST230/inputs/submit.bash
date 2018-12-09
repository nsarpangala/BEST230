for i in {1..4};
do
for j in {1..4};
do
nohup python main_script_onrate.py A1"$i""$j".txt >A1"$i""$j".out &
done
done
