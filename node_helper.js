'use strict';
const NodeHelper = require('node_helper');

const PythonShell = require('python-shell');
var pythonStarted = false

module.exports = NodeHelper.create({

	python_start: function () {
		const self = this;		

		
		this.pyshell = new PythonShell('modules/' + this.name + '/yolo-object-detector/object_detection.py', {pythonPath: 'python',  args: [JSON.stringify(this.config)]});
    		
		this.pyshell.on('message', function (message_string) {
			try{
				var message = JSON.parse(message_string)
           		//console.log("[MSG " + self.name + "] " + message);
				if (message.hasOwnProperty('status')){
					console.log("[" + self.name + "] " + message.status);
  				}
				if (message.hasOwnProperty('detected_object')){
					console.log("[" + self.name + "] detected object: " + message.detected_object.name + " with confidence " + message.detected_object.confidence + " in area "  + message.detected_object.bounds);
					self.sendSocketNotification('OD_OBJECT_FOUND', message.detected_object);
				}
				if (message.hasOwnProperty('lost_object')){
					console.log("[" + self.name + "] lost object: " + message.lost_object.name + " with confidence " + message.lost_object.confidence + " in area " + message.lost_object.bounds);
					self.sendSocketNotification('OD_OBJECT_LOST', message.lost_object);
				}
			}
			catch(err) {
				//console.log(err)
			}
   		});
  	},

  	// Subclass socketNotificationReceived received.
  	socketNotificationReceived: function(notification, payload) {
		if(notification === 'ObjectDetection_SetFPS') {
			if(pythonStarted) {
                var data = {"FPS": payload}
                this.pyshell.send(data,{mode: 'json'});

            }
        }else if(notification === 'OBJECT_DETECITON_CONFIG') {
      		this.config = payload
      		if(!pythonStarted) {
        		pythonStarted = true;
        		this.python_start();
      		};
    	};
  	}
});
