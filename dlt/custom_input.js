// requires custom_input_id predefined

// Helpers

function get_canvas_image(canvas) {
    var data = canvas.getContext("2d")
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

function set_canvas_image(canvas, values) {
    var ctx = canvas.getContext("2d");
    var data = ctx.getImageData(0, 0, canvas.width, canvas.height);
    for (var i = 0; i < values.length; ++i) {
        var v = Math.round(255 * (1 - values[i]));
        data.data[4 * i + 0] = v; // r
        data.data[4 * i + 1] = v; // g
        data.data[4 * i + 2] = v; // b
        data.data[4 * i + 3] = 255; // a
    }
    ctx.putImageData(data, 0, 0);
}

function clear_canvas(canvas) {
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Get the point on a canvas (in pixel coordinates) from a mouse/touch event
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

// Classification

function classify(strokes) {
    function handleOutput(e) {
        if (e.msg_type === "error") {
            console.warn(e.content.ename + ": " + e.content.evalue);
            console.warn(e.content.traceback);
            $(custom_input_id + "-error").text(
                e.content.ename + ": " +
                e.content.evalue + "\n" +
                e.content.traceback.join("\n").replace(/\[\d(;\d+)?m/g, ""));

        } else if (e.msg_type === "stream") {
            console.log(e.content.text);
            var output = $(custom_input_id + "-output");
            output.text(output.text() + e.content.text);

        } else if (e.msg_type === "execute_result") {
            // data comes back in JSON, but with surrounding single quotes,
            // and double-escaped
            var data = e.content.data["text/plain"].replace("\\\\", "\\");
            var d = JSON.parse(data.substring(1, data.length - 1));
            $(custom_input_id + "-label").text(d.y);
            set_canvas_image($(custom_input_id +"-output-canvas")[0], d.x);
            $(custom_input_id + "-error").text("");

        } else {
            console.error("Unknown IPython kernel message", e);
        }
    }
    $(custom_input_id + "-output").text("");
    $(custom_input_id + "-error").text("");
    var payload = {
        id: custom_input_id.substring(1),
        strokes: strokes
    };
    IPython.notebook.kernel.execute(
        "import dlt;\n" +
        "dlt.CustomInput._id_classify('" + JSON.stringify(payload) + "')",
        {iopub: {output: handleOutput}}, {silent: false});
}


// Script

var settings = {
    line_thickness: 10,
    line_reps: 1
};

var stroke_in_progress = false;
var strokes = [];
function start_stroke(e) {
    strokes.push([]);
    $.each(mouse_points(e), function(i, p) {
        strokes[strokes.length - 1].push(p);
    });
    stroke_in_progress = true;
}
function finish_stroke() {
    if (stroke_in_progress) {
        classify(strokes);
        stroke_in_progress = false;
    }
}
function move_stroke(e) {
    if (stroke_in_progress) {
        e.preventDefault();
        var points = mouse_points(e);
        $.each(points, function(i, p) {
            strokes[strokes.length - 1].push(p);
        });

        // full redraw
        var ctx = e.target.getContext("2d");
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, e.target.width, e.target.height);
        ctx.strokeStyle = "#000";
        ctx.lineWidth = settings.line_thickness;
        ctx.lineCap = "round";
        ctx.beginPath();
        $.each(strokes, function (i, stroke) {
            ctx.moveTo(stroke[0][0], stroke[0][1]);
            $.each(stroke, function (j, p) {
                ctx.lineTo(p[0], p[1]);
            });
        });
        ctx.stroke();
        return true;
    }
}
function reset_stroke() {
    strokes = [];
    stroke_in_progress = false;
    clear_canvas($(custom_input_id + "-input-canvas")[0]);
    clear_canvas($(custom_input_id + "-output-canvas")[0]);
    $(custom_input_id + "-label").text(" ");
    $(custom_input_id + "-error").text("");
    $(custom_input_id + "-output").text("");
}

reset_stroke();
$(custom_input_id + "-clear").click(reset_stroke);
$(custom_input_id + "-input-canvas")
    .mousedown(start_stroke)
    .mousemove(move_stroke)
    .mouseup(function (e) {
        if (e.which == 1) {
            finish_stroke(e);
        } else if (e.which == 2) {
            reset_stroke(e);
        }
    })
    .mouseleave(finish_stroke)
    .on("touchstart", start_stroke)
    .on("touchmove", move_stroke)
    .on("touchend", finish_stroke)
    .on("touchcancel", finish_stroke);
