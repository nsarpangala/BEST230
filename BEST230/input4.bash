ser=B43
ol=B13
for j in {1..4};
do
cp inputs/"$ol""$j".txt inputs/"$ser""$j".txt
sed -i "s/"$ol""$j"/"$ser""$j"/g" inputs/"$ser""$j".txt
sed -i "s/N,12/N,96/g" inputs/"$ser""$j".txt
done
