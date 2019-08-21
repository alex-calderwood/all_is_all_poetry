let words, coords;
let fontSize = 14;

function setup() {
    smooth();
    canvas = createCanvas(windowWidth * 2, windowWidth * 2);
    textSize(fontSize);
    textFont('Helvetica')
}

function draw() {
    let spacing = 75;
    let start = createVector(windowWidth / 2, spacing)

//    console.log(spacing * (Math.sqrt(words.length) - 1), windowWidth)
//    let bleedover = spacing * (Math.sqrt(words.length) - 1)> windowWidth
//    console.log('bleadover ' + bleedover)

    if(words) {
        for (let i = 0; i < words.length; i ++) {
            word = words[i];
            coordinates = coords[i];

            strokeWeight(10)
//            point(start.x + coordinates[0] * spacing, start.y + coordinates[1] * spacing)
            text(word, start.x + coordinates[0] * spacing, start.y + coordinates[1] * spacing);
        }
    }
    words = null;
}

function setGrid(inWords, inCoords) {
    words = JSON.parse(inWords);
    coords = JSON.parse(inCoords);
}