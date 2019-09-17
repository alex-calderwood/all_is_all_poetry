let words, coords;
let fontSize = 20;
let spacing = 100;

function setup() {
    smooth();
    canvas = createCanvas(windowWidth, windowWidth);
    textAlign(CENTER, CENTER);
    textSize(fontSize);
    textFont('Shadows Into Light')
    textFont('Josefin Slab')
}

function draw() {
    let start = createVector(windowWidth / 2, spacing)

//    console.log(spacing * (Math.sqrt(words.length) - 1), windowWidth)
//    let bleedover = spacing * (Math.sqrt(words.length) - 1)> windowWidth
//    console.log('bleadover ' + bleedover)

    if(words) {
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
    words = null;

}

function setGrid(inWords, inCoords) {
    words = JSON.parse(inWords);
    coords = JSON.parse(inCoords);
}