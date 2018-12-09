p=$1
for i in {16..60}
do
j=$(( i-1 ))
echo $j
old=AP$p$j
new=AP$p$i
cp ../subs/$old.sub ../subs/$new.sub 
sed -i "s/${old}/${new}/g" ../subs/$new.sub
sbatch ../subs/$new.sub
done
