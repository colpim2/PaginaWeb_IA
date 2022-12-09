const path = require('path')
const {createReadStream} = require('fs')
const {createServer} = require('http')
const {PORT = 3000} = process.env

const HTML_CONTENT_TYPE = 'text/html'
const CSS_CONTENT_TYPE = 'text/css'
const JS_CONTENT_TYPE = 'text/javascript'

const PUBLIC_FOLDER = path.join(__dirname)

//Definir el servidor Request - Listener
const requestListener = (req, res) => {
  const {url} = req
  let statusCode = 200
  let contentType = HTML_CONTENT_TYPE
  let stream

  //Archivos html
  if (url === '/') {
    stream = createReadStream(`templates/Landing.html`)
  } 
  //Archivos CSS
  else if (url.match("static/public/assets/css/style.css")) { 
    contentType = CSS_CONTENT_TYPE
    stream = createReadStream(`${PUBLIC_FOLDER}${url}`)
  } 
  //Archivos JavaScript
  else if (url.match("static/public/assets/js/main.js")){ 
    contentType = JS_CONTENT_TYPE
    stream = createReadStream(`${PUBLIC_FOLDER}${url}`)
  } 
  // Error
  else { 
    statusCode = 404
  }

  //Cabeceras = respuesta
  res.writeHead(statusCode, {'Content-Type': contentType})

  //If stream, enviar 
  if (stream) stream.pipe(res)

  //Else, enviar nada
  else return res.end('Not found')
}

//Crear servidor Request - Listener
const server = createServer(requestListener)

//Prender servidor
server.listen(PORT)