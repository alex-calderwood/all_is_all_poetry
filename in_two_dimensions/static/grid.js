let words, coords;
function setup() {
    smooth();
    canvas = createCanvas(windowWidth, windowHeight);
}

function draw() {
    let spacing = 35;
    let start = createVector(windowWidth / 2, spacing)
    if(words) {
        for (let i = 0; i < words.length; i ++) {
            word = words[i];
            coordinates = coords[i];

            text(word, start.x + coordinates[0] * spacing, start.y + coordinates[1] * spacing);
        }
    }
    words = null;
}

function setGrid(inWords, inCoords) {
    words = JSON.parse(inWords);
    coords = JSON.parse(inCoords);
}