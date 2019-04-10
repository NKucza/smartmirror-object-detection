/**
 * @file smartmirror-object-detection.js
 *
 * @author nkucza
 * @license MIT
 *
 * @see  https://github.com/NKucza/smartmirror-object-detection
 */

Module.register('smartmirror-object-detection',{

	defaults: {
		
	},

	start: function() {
		this.time_of_last_greeting_personal = [];
		this.time_of_last_greeting = 0;
		this.last_rec_user = [];
		this.current_user = null;
		this.sendSocketNotification('OBJECT_DETECITON_CONFIG', this.config);
		Log.info('Starting module: ' + this.name);
	},

	notificationReceived: function(notification, payload, sender) {
		if(notification === 'smartmirror-object-detectionSetFPS') {
			this.sendSocketNotification('ObjectDetection_SetFPS', payload);
        }
	},


	socketNotificationReceived: function(notification, payload) {
		if(notification === 'OD_OBJECT_FOUND') {
      
        };
	}
});
