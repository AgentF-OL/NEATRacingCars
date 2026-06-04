#Limpa pycache para garantir que carrega o novo código
rm -rf src/cars/__pycache__ src/__pycache__
#Apaga resultados antigos (fitness antiga = lixo)
rm -rf results/waypoints/* results/radars/*
