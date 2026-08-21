#!/bin/bash
set -e

SUF=$RANDOM
RG="rg-nexusops-locaweb"
LOC="germanywestcentral"
VNET="vnet-nexusops"
ST="stnexusops$SUF"
ACR="acrnexusops$SUF"
MYSQL="mysql-nexusops-$SUF"
MYSQL_DB="db_nexusops"
MYSQL_USER="adminnexus"
MYSQL_PASS="TroqueEstaSenha#2026"
LAW="law-nexusops"
AI="appi-nexusops"
ACI="aci-nexusops-api"
DNS="nexusops-api-$SUF"

echo "Criando Resource Group em $LOC..."
az group create --name $RG --location $LOC --output table

echo "Criando Virtual Network..."
az network vnet create --resource-group $RG --name $VNET --address-prefix 10.0.0.0/16 --subnet-name subnet-app --subnet-prefix 10.0.1.0/24 --output table

az network vnet subnet create --resource-group $RG --vnet-name $VNET --name subnet-data --address-prefix 10.0.2.0/24 --output table

echo "Criando Storage Account..."
az storage account create --resource-group $RG --name $ST --location $LOC --sku Standard_LRS --kind StorageV2 --output table

ST_CONN=$(az storage account show-connection-string -g $RG -n $ST --query connectionString -o tsv)

az storage container create --name raw --connection-string "$ST_CONN" --output table

az storage blob upload --container-name raw --name incidentes.csv --file incidentes.csv --connection-string "$ST_CONN" --overwrite --output table

echo "Criando MySQL..."
az mysql flexible-server create --resource-group $RG --name $MYSQL --location $LOC --admin-user $MYSQL_USER --admin-password "$MYSQL_PASS" --tier Burstable --sku-name Standard_B1ms --storage-size 20 --version 8.0.21 --public-access 0.0.0.0 --yes --output table

az mysql flexible-server db create --resource-group $RG --server-name $MYSQL --database-name $MYSQL_DB --output table

MYSQL_HOST=$(az mysql flexible-server show -g $RG -n $MYSQL --query fullyQualifiedDomainName -o tsv)
echo "✓ Host MySQL: $MYSQL_HOST"

MEU_IP=$(curl -s https://api.ipify.org)
az mysql flexible-server firewall-rule create --resource-group $RG --name $MYSQL --rule-name meu-ip --start-ip-address $MEU_IP --end-ip-address $MEU_IP --output table

echo "Criando Log Analytics..."
az monitor log-analytics workspace create --resource-group $RG --workspace-name $LAW --location $LOC --output table

LAW_ID=$(az monitor log-analytics workspace show -g $RG -n $LAW --query customerId -o tsv)
LAW_KEY=$(az monitor log-analytics workspace get-shared-keys -g $RG -n $LAW --query primarySharedKey -o tsv)
LAW_RESOURCE=$(az monitor log-analytics workspace show -g $RG -n $LAW --query id -o tsv)

echo "Criando Application Insights..."
az extension add --name application-insights --yes 2>/dev/null || true
az monitor app-insights component create --app $AI --location $LOC --resource-group $RG --workspace "$LAW_RESOURCE" --output table

AI_CONN=$(az monitor app-insights component show --app $AI -g $RG --query connectionString -o tsv)

echo "Criando Container Registry..."
az acr create --resource-group $RG --name $ACR --sku Basic --admin-enabled true --output table

echo "Buildando imagem (pode levar 3-5 min)..."
az acr build --registry $ACR --image api-previsao:v1 . --output table

ACR_USER=$(az acr credential show -n $ACR --query username -o tsv)
ACR_PASS=$(az acr credential show -n $ACR --query "passwords[0].value" -o tsv)

echo "Subindo container no ACI..."
az container create --resource-group $RG --name $ACI --image $ACR.azurecr.io/api-previsao:v1 --registry-login-server $ACR.azurecr.io --registry-username $ACR_USER --registry-password "$ACR_PASS" --cpu 1 --memory 1.5 --os-type Linux --ports 80 --dns-name-label $DNS --environment-variables MYSQL_HOST=$MYSQL_HOST MYSQL_USER=$MYSQL_USER MYSQL_DB=$MYSQL_DB BLOB_CONTAINER=raw BLOB_ARQUIVO=incidentes.csv --secure-environment-variables MYSQL_PASSWORD="$MYSQL_PASS" AZURE_STORAGE_CONNECTION_STRING="$ST_CONN" APPLICATIONINSIGHTS_CONNECTION_STRING="$AI_CONN" --log-analytics-workspace $LAW_ID --log-analytics-workspace-key $LAW_KEY --restart-policy OnFailure --output table

echo ""
echo "PROVISIONAMENTO CONCLUÍDO!"
URL="http://$DNS.germanywestcentral.azurecontainer.io"
echo "Aplicação em: $URL"
echo "Swagger em: $URL/docs"
echo ""
echo "Aguardando 60 segundos para API ficar pronta..."
sleep 60

echo "Testando endpoints..."
curl -s $URL/ | jq . 2>/dev/null || echo "API ainda inicializando"
curl -s -X POST $URL/ingerir | jq . 2>/dev/null || echo "Teste /ingerir"
curl -s -X POST $URL/prever | jq . 2>/dev/null || echo "Teste /prever"
