let words, coords;
let fontSize = 20;
let spacing = 250;
let a = 0;

let doDraw = false;

function setup() {
    smooth();
    canvas = createCanvas(windowWidth, windowWidth);
    textAlign(CENTER, CENTER);
    textSize(fontSize);
}

function draw() {
    let start = createVector(windowWidth / 2 - spacing / 2, windowHeight / 8)

//    console.log(spacing * (Math.sqrt(words.length) - 1), windowWidth)
//    let bleedover = spacing * (Math.sqrt(words.length) - 1)> windowWidth
//    console.log('bleadover ' + bleedover)
    textFont('Josefin Slab')

    if(doDraw) {
        background(255);

        console.log('angle', a);
        X = []
        Y = []
        for (let c of coords) {
            X.push(c[0]);
            Y.push(c[1]);
        }
        console.log('X', Math.min(...X), Math.max(...X))
        console.log('Y', Math.min(...Y), Math.max(...Y))

        for (let i = 0; i < words.length; i ++) {
            word = words[i];
            coordinates = coords[i];

            let x = start.x + coordinates[0] * spacing;
            let y = start.y + coordinates[1] * spacing;

            // if (y > maxY) {
            //     maxY = y;
            //     print('resize', y, true)
            //     resizeCanvas(windowWidth, maxY + 10);
            // }
            // strokeWeight(10)
            // point(x, y)

            text(word, x, y);
        }
    }
    doDraw = false;
}

function setGrid(inWords, inCoords) {
    words = JSON.parse(inWords);
    coords = JSON.parse(inCoords);
    doDraw = true;
}