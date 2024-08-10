# React UI

This project uses React to create a simple UI. 
The UI is based on ant design a popular react UI library.
The compiler is Vite, a modern build tool that is faster than webpack.

## How to run the project

### Linter

Run `npm run lint` to run the linter, this is required as if the linter fails the build will fail as well.
Note: This step is not required if you are using the docker container provided and are not making any changes to the code.

### Development

1. Clone the project
2. Run `npm install`
3. Run `npm run dev`

### Docker

1. Clone the project
2. Run `docker compose up` in the root directory
3. Open `http://localhost:8080` in your browser

## The Flask API

A small flask API is used to serve the data to the UI, it serves as an example of how to use a backend with a react frontend.
The conversations are not stored in a database, they are stored in memory and are lost when the server is restarted.

The API can be started by running `python ./Main.py` in the `FlaskServer` directory.

To implement another api, you can reuse directly the same endpoints and it will integrate directly with the UI.

## Environment variables

`
VITE_BACKEND_URL="http://localhost:5000" {The URL of the backend API}
VITE_IMG_LOGO="dist/GPT_Tator.jfif" {The URL of the logo image}
VITE_IMG_LOGO_ALT="GPT Tator Logo" {The alt text of the logo image}
`
