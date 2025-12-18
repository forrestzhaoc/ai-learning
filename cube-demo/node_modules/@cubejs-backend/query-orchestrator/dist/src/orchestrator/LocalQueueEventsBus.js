"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LocalQueueEventsBus = void 0;
const BaseQueueEventsBus_1 = require("./BaseQueueEventsBus");
class LocalQueueEventsBus extends BaseQueueEventsBus_1.BaseQueueEventsBus {
    emit(event) {
        Promise.all(Object.values(this.subscribers).map(({ callback }) => callback(event)))
            .catch(err => console.error(err));
    }
}
exports.LocalQueueEventsBus = LocalQueueEventsBus;
//# sourceMappingURL=LocalQueueEventsBus.js.map