ser=A


for i in {1..4};
do
for j in {1..4};
do
cp subs/A1"$i""$j".sub subs/A4"$i""$j".txt
sed -i "s/A1"$i""$j"/A4"$i""$j"/g" inputs/A4"$i""$j".txt
sed -i "s/N,12/N,96/g" inputs/A4"$i""$j".txt
