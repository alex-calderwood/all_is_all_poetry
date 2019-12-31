let words, coords;
let fontSize = 20;
let defaultSpacing = 90;
let sapcing = null;

let realCanvasY = 100;

let drawCount = 0;
let n;

function setup() {
    smooth();
    canvas = createCanvas(windowWidth, windowWidth * 3/2);
    canvas.position(0, 0)
    canvas.style('position', 'absolute')
    canvas.style('z-index', '-1')
    textAlign(CENTER, CENTER);
    textSize(fontSize);
    frameRate(32);

    up = select('#up');
    down = select('#down');
    left = select('#left');
    right = select('#right');

    noStroke()
}

function draw() {

    let start = createVector(windowWidth / 2 - spacing / 2, realCanvasY + windowHeight / 6)

    textFont('Josefin Slab')

    if (drawCount == 0) {
        fill(255)
        rect(0, 200, windowWidth, windowWidth * 3 /2);
    } else {
        fill(255, 255, 255, 30)
        rect(0, realCanvasY, windowWidth, windowWidth * 3 /2);
    }

    // clear()

    up.position(windowWidth / 2)


    fill(0, 0, 0, 40);

    // let noise = max(0, random((300 - drawCount)));
    let window = 60;

    for (let i = 0; i < words.length; i ++) {
        word = words[i];
        coordinates = coords[i];

        let w =  max(0, window - drawCount)
        let noiseX = w * random() - w / 2;
        let noiseY = w * random() - w / 2;

        let x = start.x + coordinates[0] * spacing;
        let y = start.y + coordinates[1] * spacing;

        strokeWeight(1)
        text(word, noiseX + x, noiseY +  y);

        if (i == 0) {
            up.position(x - 100, y - 100 - 15);
        } else if (i == n * n - 1) {
            down.position(x - 100, y + 100 - 15);
        } else if (i == n - 1) {
            left.position(x - 200 - 100, y - 15);
        }  else if (i == n * n - n) {
            right.position(x + 100, y - 15);
        }

        // For debugging text position
        // strokeWeight(10);
        // point(x, y);
    }

    drawCount += 1;
}

function setGrid(inWords, inCoords) {
    words = JSON.parse(inWords);
    coords = JSON.parse(inCoords);
    spacing = defaultSpacing * Math.log2(2 * coords.length);
    drawCount = 0;
    n = Math.sqrt(words.length)
}