ser=B13
for j in {1..4};
do
cp inputs/B11"$j".txt inputs/"$ser""$j".txt
sed -i "s/B11"$j"/"$ser""$j"/g" inputs/"$ser""$j".txt
sed -i "s/D,0.13/D,13/g" inputs/"$ser""$j".txt
sed -i "s/dt,1e-4/dt,1e-7/g" inputs/"$ser""$j".txt
done
