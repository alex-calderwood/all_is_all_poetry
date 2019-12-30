let words, coords;
let fontSize = 20;
let defaultSpacing = 90;
let sapcing = null;

let drawCount = 0;

function setup() {
    smooth();
    canvas = createCanvas(windowWidth, windowWidth * 3/2);
    textAlign(CENTER, CENTER);
    textSize(fontSize);
    frameRate(32);
    fill(0, 0, 0, 15);
}

function draw() {
    let start = createVector(windowWidth / 2 - spacing / 2, windowHeight / 8)

    textFont('Josefin Slab')

    if (drawCount == 0) {
        background(255);
    } else {
        background(255, 255, 255, 5);
    }

    // let noise = max(0, random((300 - drawCount)));
    let window = 60;

    for (let i = 0; i < words.length; i ++) {
        word = words[i];
        coordinates = coords[i];

        let noiseX = max(0, (window - drawCount) / 2) * random() - (window);
        let noiseY = max(0, (window - drawCount) / 2) * random() - (window);

        let x = noiseX + start.x + coordinates[0] * spacing;
        let y = noiseY + start.y + coordinates[1] * spacing;

        text(word, x, y);
    }

    drawCount += 1;
}

function setGrid(inWords, inCoords) {
    words = JSON.parse(inWords);
    coords = JSON.parse(inCoords);
    spacing = defaultSpacing * Math.log2(2 * coords.length);
    drawCount = 0;
}