ser=B42
for j in {1..4};
do
cp inputs/B12"$j".txt inputs/"$ser""$j".txt
sed -i "s/B12"$j"/"$ser""$j"/g" inputs/"$ser""$j".txt
sed -i "s/N,12/N,96/g" inputs/"$ser""$j".txt
done
