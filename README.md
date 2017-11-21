# NaoAI

Installing Nao python SDK
1. [Download nao python sdk and c++ sdk](https://community.ald.softbankrobotics.com/en/resources/software/language/en-gb/robot/nao-2]). You will need c++ sdk for boostlibs.    
2. Extract both and move python sdk to your favorite place.   
3. 
```
export PYTHONPATH=${PYTHONPATH}:/path/to/nao-python-sdk/
export LD_LIBRARY_PATH="/path/to/nao-python-sdk:$LD_LIBRARY_PATH"
sudo ldconfig
sudo cp -a /path/to/naoqicpp/lib/libboost_*.so.* /path/to/nao-python-sdk/
```
