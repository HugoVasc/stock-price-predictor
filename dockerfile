FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

COPY 'src/model/lstm_stock_data_model.pth' .

COPY 'src/api/' .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Definir a variável de ambiente para o handler do Lambda
ENV AWS_LAMBDA_HANDLER="main.handler"

# Instalar o AWS Lambda Runtime Interface
RUN pip install awslambdaric

# Comando para rodar a aplicação no AWS Lambda
CMD ["python", "-m", "awslambdaric"]
