copy the project to server
```
scp -r /Users/jowonkim/Documents/GitHub/Masterthesis  zzhang4@snellius.surf.nl:/home/zzhang4/

```
copy the result
```
scp zzhang4@snellius.surf.nl:/home/zzhang4/Masterthesis/model/model_full.pth /Users/jowonkim/Documents/GitHub/Masterthesis/model 


```

Run the Job in a SLURM Cluster
```
sbatch training/jobs/supervised_train.job
```

monitor job with:
```
squeue -u $USER
```

```
ls -lh train-*.out

tail -f train-10256766.out
```
cancel job
```
 scancel 10256882
```

Check for Active Processes
```
top -u zzhang4
```

check job
```
scontrol show job 10257384
```
remove 
```
rm -rf Masterthesis
```