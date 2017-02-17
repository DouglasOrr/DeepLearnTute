// custom_input_id

var settings = {
    line_thickness: 1,
    line_reps: 2
};

function get_canvas_image() {
    var canvas = $(custom_input_id + "-canvas")[0];
    var data = canvas.getContext('2d')
        .getImageData(0, 0, canvas.width, canvas.height)
        .data;
    var result = new Array(canvas.width * canvas.height);
    for (var i = 0; i < result.length; ++i) {
        var r = data[4 * i + 0] / 255;
        var g = data[4 * i + 1] / 255;
        var b = data[4 * i + 2] / 255;
        var a = data[4 * i + 3] / 255;
        result[i] = a * (1 - r) * (1 - g) * (1 - b);
    }
    return result;
}

function clear_canvas_image() {
    var canvas = $(custom_input_id + "-canvas")[0];
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function classify() {
    function handleOutput(e) {
        if (e.msg_type === "error") {
            console.log(e.content.ename);
            console.log(e.content.evalue);

        } else if (e.msg_type === "stream") {
            console.log(e.content.text);

        } else if (e.msg_type === "execute_result") {
            // data comes back with surrounding single quotes
            var label = e.content.data['text/plain']
                .replace(new RegExp("'", 'g'), '');
            $(custom_input_id + "-label").text(label);

        } else {
            console.error("Unknown IPython kernel message", e);
        }
    }
    var payload = {
        id: custom_input_id.substring(1),
        data: get_canvas_image()
    };
    IPython.notebook.kernel.execute(
        "import dlt;\n" +
        "dlt.CustomInput._id_classify('" + JSON.stringify(payload) + "')",
        {iopub: {output: handleOutput}}, {silent: false});
}


// Handlers

clear_canvas_image();

$(custom_input_id + "-classify").click(classify);
$(custom_input_id + "-clear").click(clear_canvas_image);

// Get the point on a canvas (in pixel coordinates) from mouse
function mouse_points(e) {
    var xscale = e.target.clientWidth / e.target.width;
    var yscale = e.target.clientHeight / e.target.height;
    if (e.offsetX !== undefined) {
        // mouse event
        return [[e.offsetX / xscale, e.offsetY / yscale]];
    } else {
        // touch event
        var points = [];
        $.each(e.originalEvent.changedTouches, function (i, o) {
            var parent = o.target.getBoundingClientRect();
            var x = o.clientX - parent.left;
            var y = o.clientY - parent.top;
            points.push([x / xscale, y / yscale])
        });
        return points;
    }
}

var last_point = null;
function finish_trace() {
    if (last_point !== null) {
        classify();
    }
    last_point = null;
}
function move_trace(e) {
    if (last_point !== null) {
        e.preventDefault();
        var points = mouse_points(e);
        var ctx = e.target.getContext('2d');
        ctx.strokeStyle = "#000";
        ctx.lineWidth = settings.line_thickness;
        ctx.beginPath();
        ctx.moveTo(last_point[0], last_point[1]);
        $.each(points, function (i, p) {
            ctx.lineTo(p[0], p[1]);
        });
        // Draw multiple times to make the line darker
        // (fool antialiasing a bit)
        for (var i = 0; i < settings.line_reps; ++i) {
            ctx.stroke();
        }
        last_point = points[points.length - 1];
        return true;
    }
}
$(custom_input_id + "-canvas")
    .mousedown(function (e) { last_point = mouse_points(e)[0]; })
    .mousemove(move_trace)
    .mouseup(finish_trace)
    .mouseleave(finish_trace)
    .on('touchstart', function (e) { last_point = mouse_points(e)[0]; })
    .on('touchmove', move_trace)
    .on('touchend', finish_trace)
    .on('touchcancel', finish_trace);
