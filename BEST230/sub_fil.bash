ser=B43
for j in {1..4};
do
cp subs/B41"$j".sub subs/"$ser""$j".sub
sed -i "s/B41"$j"/"$ser""$j"/g" subs/"$ser""$j".sub
done
