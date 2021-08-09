FROM node:14
WORKDIR /usr/src/app
COPY package*.json ./
RUN npm install
RUN npm install --global serve
COPY . .
EXPOSE 8080
CMD [ "serve", "-s", "build", "-l", "8080" ]
