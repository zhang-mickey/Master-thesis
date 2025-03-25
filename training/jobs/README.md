## How to run the job

copy the project to server
```
scp -r /Users/jowonkim/Documents/GitHub/Masterthesis  zzhang4@snellius.surf.nl:/home/zzhang4/

```
copy the result
```
scp zzhang4@snellius.surf.nl:/home/zzhang4/Masterthesis/model/deeplabv3plus_resnet50_model_full.pth /Users/jowonkim/Documents/GitHub/Masterthesis/model 


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
check the storage
```
prjspc-quota /projects/0/prjs1418
/gpfs/work5/0/prjs1418|TiB:quota=1.00,limit=1.00,usage=0.0000%|Inodes:quota=1000000,limit=1100000,usage=0.0001%
```