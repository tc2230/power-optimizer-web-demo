# 指定基礎Image(若需要其他用途的image可由此搜尋: https://hub.docker.com/)
FROM python:3.11-slim

# 在根目錄建立 mnt 資料夾
RUN mkdir /init

# 將 Dockerfile 同一目錄下的所有檔案(包含 requirements.txt 和 main.py) 複製到 container 中的 /code 資料夾
## COPY <source_file_path> <destination_file_path>
COPY ./* /init

# 在建立 container 時執行 "pip install" 指令，安裝 requirement.txt 中的內容
RUN pip install --no-cache-dir --upgrade -r /init/requirements.txt
#RUN pip install --force-reinstall 'requests<2.29.0' 'urllib3<2.0'

# 也可以直接把要安裝的工具寫在Dockerfile中
# EX:
# RUN pip install pandas

# 在 container 內安裝 nano editor, crontab
RUN apt-get update && apt-get install nano

# 改變工作目錄
WORKDIR /init


# container 被開啟時，預設要執行的指令
# 若需要設定host ip, 可加入"--ip <host_ip>" (預設為127.0.0.1)
# 若需要設定port, 可加入"--port <port>" (預設為8888)
# 若需要設定開啟時的指定目錄, 可加入"--ServerApp.root_dir <path>"
# 若需要設定token, 可加入"--ServerApp.token "<your_token>", (若未新增, JupyterLab會隨機產生一組)
# CMD ["nohup", "streamlit", "run", "web_demo.py", "--server.port", "8080", "&"]


#### docker 指令
# 1. 移至Dockerfile所在目錄, 並將要包進Image的檔案也都移至相同目錄下

# 2. docker build --tag <image_name>:<tag> <working_directory>
# Ex: docker build --tag bi_test:v0 .

# 3. docker run -dit --name <容器名字> -p <主機port>:<容器內部port> -v <外部實體路徑>:<container內部路徑> -w <進入container時的工作路徑> <image_name>:<tag>
# Ex: docker run -dit --name bi_test -p 7777:9527 -v /mnt/c/Users/taylor.fu/Desktop/工作資料/bi_python_tutorial/tmp/:/mnt/ -w /mnt bi_test:v0