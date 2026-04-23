#№1. download
python3 -m venv ./my_env
. ./my_env/bin/activate
python3 -m ensurepip --upgrade
pip3 install setuptools
pip3 install -r requirements.txt
python3 download.py
#-----------------------

#№2. train_model 
echo "Start train model"
cd /var/lib/jenkins/workspace/Download/
. ./my_env/bin/activate
python3 train_model.py > best_model.txt
#------------------------

#3. deploy 
cd /var/lib/jenkins/workspace/Download/
. ./my_env/bin/activate
export BUILD_ID=dontKillMe
export JENKINS_NODE_COOKIE=dontKillMe
path_model=$(cat best_model.txt)
mlflow models serve -m $path_model -p 5003 --env-manager local &
#------------------------

#4. healthy (status service)
curl -X POST http://127.0.0.1:5003/invocations \
     -H 'Content-Type: application/json' \
     -d '{"inputs": [[0.0, 1.0, 1.0, 1.0, 72, 72, 74]]}'


#Pipeline - для объедения задач в последовательный конвеер
pipeline {
    agent any

    stages {
        stage('Start Download') {
            steps {
                
                build job: "download"
                
            }
        }
        
        stage ('Train') {
            
            steps {
                
                script {
                    dir('/var/lib/jenkins/workspace/download') {
                        build job: "train model"
                    }
                }
            
            }
        }
        
        stage ('Deploy') {
            steps {
                build job: 'deploy'
            }
        }
        
        stage ('Status') {
            steps {
                build job: 'healthy'
            }
        }
    }
}
