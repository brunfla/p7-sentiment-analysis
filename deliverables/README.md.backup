## DVC
```
dvc init
```

### Pour un stockage local (exemple : dans un répertoire spécifique en dehors de votre projet) :
```
dvc remote add -d myremote /path/to/storage
```
### Pour un bucket S3 :
```
dvc remote add -d myremote s3://p7-sentiment-analysis

dvc remote modify myremote access_key_id admin
dvc remote modify myremote secret_access_key XKzq2CuGKQ  
dvc remote modify myremote endpointurl  http://api.minio.local 

```

```
touch data/output/clean_data.csv
(mlops-env) ➜  p7-sentiment-analysis git:(main) ✗ dvc add data/output/clean_data.csv
100% Adding...|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████|1/1 [00:02,  2.89s/file]

```

###
```
kubectl patch svc mlflow-minio -p '{"spec": {"type": "LoadBalancer"}}'
```

dvc push
