/*eslint-disable block-scoped-var, id-length, no-control-regex, no-magic-numbers, no-prototype-builtins, no-redeclare, no-shadow, no-var, sort-vars*/
import * as $protobuf from "protobufjs/minimal";

// Common aliases
const $Reader = $protobuf.Reader, $Writer = $protobuf.Writer, $util = $protobuf.util;

// Exported root namespace
const $root = $protobuf.roots["default"] || ($protobuf.roots["default"] = {});

export const lagrangian = $root.lagrangian = (() => {

    /**
     * Namespace lagrangian.
     * @exports lagrangian
     * @namespace
     */
    const lagrangian = {};

    lagrangian.testing = (function() {

        /**
         * Namespace testing.
         * @memberof lagrangian
         * @namespace
         */
        const testing = {};

        testing.ModelForward = (function() {

            /**
             * Properties of a ModelForward.
             * @memberof lagrangian.testing
             * @interface IModelForward
             * @property {number|Long|null} [nbody] ModelForward nbody
             * @property {Array.<number>|null} [xpos] ModelForward xpos
             * @property {Array.<number>|null} [xquat] ModelForward xquat
             */

            /**
             * Constructs a new ModelForward.
             * @memberof lagrangian.testing
             * @classdesc Represents a ModelForward.
             * @implements IModelForward
             * @constructor
             * @param {lagrangian.testing.IModelForward=} [properties] Properties to set
             */
            function ModelForward(properties) {
                this.xpos = [];
                this.xquat = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null && keys[i] !== "__proto__")
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * ModelForward nbody.
             * @member {number|Long} nbody
             * @memberof lagrangian.testing.ModelForward
             * @instance
             */
            ModelForward.prototype.nbody = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * ModelForward xpos.
             * @member {Array.<number>} xpos
             * @memberof lagrangian.testing.ModelForward
             * @instance
             */
            ModelForward.prototype.xpos = $util.emptyArray;

            /**
             * ModelForward xquat.
             * @member {Array.<number>} xquat
             * @memberof lagrangian.testing.ModelForward
             * @instance
             */
            ModelForward.prototype.xquat = $util.emptyArray;

            /**
             * Creates a new ModelForward instance using the specified properties.
             * @function create
             * @memberof lagrangian.testing.ModelForward
             * @static
             * @param {lagrangian.testing.IModelForward=} [properties] Properties to set
             * @returns {lagrangian.testing.ModelForward} ModelForward instance
             */
            ModelForward.create = function create(properties) {
                return new ModelForward(properties);
            };

            /**
             * Encodes the specified ModelForward message. Does not implicitly {@link lagrangian.testing.ModelForward.verify|verify} messages.
             * @function encode
             * @memberof lagrangian.testing.ModelForward
             * @static
             * @param {lagrangian.testing.IModelForward} message ModelForward message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ModelForward.encode = function encode(message, writer, q) {
                if (!writer)
                    writer = $Writer.create();
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                if (message.nbody != null && Object.hasOwnProperty.call(message, "nbody"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int64(message.nbody);
                if (message.xpos != null && message.xpos.length) {
                    writer.uint32(/* id 2, wireType 2 =*/18).fork();
                    for (let i = 0; i < message.xpos.length; ++i)
                        writer.double(message.xpos[i]);
                    writer.ldelim();
                }
                if (message.xquat != null && message.xquat.length) {
                    writer.uint32(/* id 3, wireType 2 =*/26).fork();
                    for (let i = 0; i < message.xquat.length; ++i)
                        writer.double(message.xquat[i]);
                    writer.ldelim();
                }
                return writer;
            };

            /**
             * Encodes the specified ModelForward message, length delimited. Does not implicitly {@link lagrangian.testing.ModelForward.verify|verify} messages.
             * @function encodeDelimited
             * @memberof lagrangian.testing.ModelForward
             * @static
             * @param {lagrangian.testing.IModelForward} message ModelForward message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            ModelForward.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer && writer.len ? writer.fork() : writer).ldelim();
            };

            /**
             * Decodes a ModelForward message from the specified reader or buffer.
             * @function decode
             * @memberof lagrangian.testing.ModelForward
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {lagrangian.testing.ModelForward} ModelForward
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ModelForward.decode = function decode(reader, length, error, long) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                if (long === undefined)
                    long = 0;
                if (long > $Reader.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.lagrangian.testing.ModelForward();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.nbody = reader.int64();
                            break;
                        }
                    case 2: {
                            if (!(message.xpos && message.xpos.length))
                                message.xpos = [];
                            if ((tag & 7) === 2) {
                                let end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.xpos.push(reader.double());
                            } else
                                message.xpos.push(reader.double());
                            break;
                        }
                    case 3: {
                            if (!(message.xquat && message.xquat.length))
                                message.xquat = [];
                            if ((tag & 7) === 2) {
                                let end2 = reader.uint32() + reader.pos;
                                while (reader.pos < end2)
                                    message.xquat.push(reader.double());
                            } else
                                message.xquat.push(reader.double());
                            break;
                        }
                    default:
                        reader.skipType(tag & 7, long);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a ModelForward message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof lagrangian.testing.ModelForward
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {lagrangian.testing.ModelForward} ModelForward
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            ModelForward.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a ModelForward message.
             * @function verify
             * @memberof lagrangian.testing.ModelForward
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            ModelForward.verify = function verify(message, long) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    return "maximum nesting depth exceeded";
                if (message.nbody != null && Object.hasOwnProperty.call(message, "nbody"))
                    if (!$util.isInteger(message.nbody) && !(message.nbody && $util.isInteger(message.nbody.low) && $util.isInteger(message.nbody.high)))
                        return "nbody: integer|Long expected";
                if (message.xpos != null && Object.hasOwnProperty.call(message, "xpos")) {
                    if (!Array.isArray(message.xpos))
                        return "xpos: array expected";
                    for (let i = 0; i < message.xpos.length; ++i)
                        if (typeof message.xpos[i] !== "number")
                            return "xpos: number[] expected";
                }
                if (message.xquat != null && Object.hasOwnProperty.call(message, "xquat")) {
                    if (!Array.isArray(message.xquat))
                        return "xquat: array expected";
                    for (let i = 0; i < message.xquat.length; ++i)
                        if (typeof message.xquat[i] !== "number")
                            return "xquat: number[] expected";
                }
                return null;
            };

            /**
             * Creates a ModelForward message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof lagrangian.testing.ModelForward
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {lagrangian.testing.ModelForward} ModelForward
             */
            ModelForward.fromObject = function fromObject(object, long) {
                if (object instanceof $root.lagrangian.testing.ModelForward)
                    return object;
                if (!$util.isObject(object))
                    throw TypeError(".lagrangian.testing.ModelForward: object expected");
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let message = new $root.lagrangian.testing.ModelForward();
                if (object.nbody != null)
                    if ($util.Long)
                        message.nbody = $util.Long.fromValue(object.nbody, false);
                    else if (typeof object.nbody === "string")
                        message.nbody = parseInt(object.nbody, 10);
                    else if (typeof object.nbody === "number")
                        message.nbody = object.nbody;
                    else if (typeof object.nbody === "object")
                        message.nbody = new $util.LongBits(object.nbody.low >>> 0, object.nbody.high >>> 0).toNumber();
                if (object.xpos) {
                    if (!Array.isArray(object.xpos))
                        throw TypeError(".lagrangian.testing.ModelForward.xpos: array expected");
                    message.xpos = [];
                    for (let i = 0; i < object.xpos.length; ++i)
                        message.xpos[i] = Number(object.xpos[i]);
                }
                if (object.xquat) {
                    if (!Array.isArray(object.xquat))
                        throw TypeError(".lagrangian.testing.ModelForward.xquat: array expected");
                    message.xquat = [];
                    for (let i = 0; i < object.xquat.length; ++i)
                        message.xquat[i] = Number(object.xquat[i]);
                }
                return message;
            };

            /**
             * Creates a plain object from a ModelForward message. Also converts values to other types if specified.
             * @function toObject
             * @memberof lagrangian.testing.ModelForward
             * @static
             * @param {lagrangian.testing.ModelForward} message ModelForward
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            ModelForward.toObject = function toObject(message, options, q) {
                if (!options)
                    options = {};
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                let object = {};
                if (options.arrays || options.defaults) {
                    object.xpos = [];
                    object.xquat = [];
                }
                if (options.defaults)
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.nbody = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : typeof BigInt !== "undefined" && options.longs === BigInt ? long.toBigInt() : long;
                    } else
                        object.nbody = options.longs === String ? "0" : typeof BigInt !== "undefined" && options.longs === BigInt ? BigInt("0") : 0;
                if (message.nbody != null && Object.hasOwnProperty.call(message, "nbody"))
                    if (typeof BigInt !== "undefined" && options.longs === BigInt)
                        object.nbody = typeof message.nbody === "number" ? BigInt(message.nbody) : $util.Long.fromBits(message.nbody.low >>> 0, message.nbody.high >>> 0, false).toBigInt();
                    else if (typeof message.nbody === "number")
                        object.nbody = options.longs === String ? String(message.nbody) : message.nbody;
                    else
                        object.nbody = options.longs === String ? $util.Long.prototype.toString.call(message.nbody) : options.longs === Number ? new $util.LongBits(message.nbody.low >>> 0, message.nbody.high >>> 0).toNumber() : message.nbody;
                if (message.xpos && message.xpos.length) {
                    object.xpos = [];
                    for (let j = 0; j < message.xpos.length; ++j)
                        object.xpos[j] = options.json && !isFinite(message.xpos[j]) ? String(message.xpos[j]) : message.xpos[j];
                }
                if (message.xquat && message.xquat.length) {
                    object.xquat = [];
                    for (let j = 0; j < message.xquat.length; ++j)
                        object.xquat[j] = options.json && !isFinite(message.xquat[j]) ? String(message.xquat[j]) : message.xquat[j];
                }
                return object;
            };

            /**
             * Converts this ModelForward to JSON.
             * @function toJSON
             * @memberof lagrangian.testing.ModelForward
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            ModelForward.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for ModelForward
             * @function getTypeUrl
             * @memberof lagrangian.testing.ModelForward
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            ModelForward.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/lagrangian.testing.ModelForward";
            };

            return ModelForward;
        })();

        testing.EnvUpdates = (function() {

            /**
             * Properties of an EnvUpdates.
             * @memberof lagrangian.testing
             * @interface IEnvUpdates
             * @property {Array.<lagrangian.testing.IModelForward>|null} [updates] EnvUpdates updates
             */

            /**
             * Constructs a new EnvUpdates.
             * @memberof lagrangian.testing
             * @classdesc Represents an EnvUpdates.
             * @implements IEnvUpdates
             * @constructor
             * @param {lagrangian.testing.IEnvUpdates=} [properties] Properties to set
             */
            function EnvUpdates(properties) {
                this.updates = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null && keys[i] !== "__proto__")
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * EnvUpdates updates.
             * @member {Array.<lagrangian.testing.IModelForward>} updates
             * @memberof lagrangian.testing.EnvUpdates
             * @instance
             */
            EnvUpdates.prototype.updates = $util.emptyArray;

            /**
             * Creates a new EnvUpdates instance using the specified properties.
             * @function create
             * @memberof lagrangian.testing.EnvUpdates
             * @static
             * @param {lagrangian.testing.IEnvUpdates=} [properties] Properties to set
             * @returns {lagrangian.testing.EnvUpdates} EnvUpdates instance
             */
            EnvUpdates.create = function create(properties) {
                return new EnvUpdates(properties);
            };

            /**
             * Encodes the specified EnvUpdates message. Does not implicitly {@link lagrangian.testing.EnvUpdates.verify|verify} messages.
             * @function encode
             * @memberof lagrangian.testing.EnvUpdates
             * @static
             * @param {lagrangian.testing.IEnvUpdates} message EnvUpdates message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            EnvUpdates.encode = function encode(message, writer, q) {
                if (!writer)
                    writer = $Writer.create();
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                if (message.updates != null && message.updates.length)
                    for (let i = 0; i < message.updates.length; ++i)
                        $root.lagrangian.testing.ModelForward.encode(message.updates[i], writer.uint32(/* id 1, wireType 2 =*/10).fork(), q + 1).ldelim();
                return writer;
            };

            /**
             * Encodes the specified EnvUpdates message, length delimited. Does not implicitly {@link lagrangian.testing.EnvUpdates.verify|verify} messages.
             * @function encodeDelimited
             * @memberof lagrangian.testing.EnvUpdates
             * @static
             * @param {lagrangian.testing.IEnvUpdates} message EnvUpdates message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            EnvUpdates.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer && writer.len ? writer.fork() : writer).ldelim();
            };

            /**
             * Decodes an EnvUpdates message from the specified reader or buffer.
             * @function decode
             * @memberof lagrangian.testing.EnvUpdates
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {lagrangian.testing.EnvUpdates} EnvUpdates
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            EnvUpdates.decode = function decode(reader, length, error, long) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                if (long === undefined)
                    long = 0;
                if (long > $Reader.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.lagrangian.testing.EnvUpdates();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            if (!(message.updates && message.updates.length))
                                message.updates = [];
                            message.updates.push($root.lagrangian.testing.ModelForward.decode(reader, reader.uint32(), undefined, long + 1));
                            break;
                        }
                    default:
                        reader.skipType(tag & 7, long);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes an EnvUpdates message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof lagrangian.testing.EnvUpdates
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {lagrangian.testing.EnvUpdates} EnvUpdates
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            EnvUpdates.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies an EnvUpdates message.
             * @function verify
             * @memberof lagrangian.testing.EnvUpdates
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            EnvUpdates.verify = function verify(message, long) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    return "maximum nesting depth exceeded";
                if (message.updates != null && Object.hasOwnProperty.call(message, "updates")) {
                    if (!Array.isArray(message.updates))
                        return "updates: array expected";
                    for (let i = 0; i < message.updates.length; ++i) {
                        let error = $root.lagrangian.testing.ModelForward.verify(message.updates[i], long + 1);
                        if (error)
                            return "updates." + error;
                    }
                }
                return null;
            };

            /**
             * Creates an EnvUpdates message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof lagrangian.testing.EnvUpdates
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {lagrangian.testing.EnvUpdates} EnvUpdates
             */
            EnvUpdates.fromObject = function fromObject(object, long) {
                if (object instanceof $root.lagrangian.testing.EnvUpdates)
                    return object;
                if (!$util.isObject(object))
                    throw TypeError(".lagrangian.testing.EnvUpdates: object expected");
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let message = new $root.lagrangian.testing.EnvUpdates();
                if (object.updates) {
                    if (!Array.isArray(object.updates))
                        throw TypeError(".lagrangian.testing.EnvUpdates.updates: array expected");
                    message.updates = [];
                    for (let i = 0; i < object.updates.length; ++i) {
                        if (!$util.isObject(object.updates[i]))
                            throw TypeError(".lagrangian.testing.EnvUpdates.updates: object expected");
                        message.updates[i] = $root.lagrangian.testing.ModelForward.fromObject(object.updates[i], long + 1);
                    }
                }
                return message;
            };

            /**
             * Creates a plain object from an EnvUpdates message. Also converts values to other types if specified.
             * @function toObject
             * @memberof lagrangian.testing.EnvUpdates
             * @static
             * @param {lagrangian.testing.EnvUpdates} message EnvUpdates
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            EnvUpdates.toObject = function toObject(message, options, q) {
                if (!options)
                    options = {};
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                let object = {};
                if (options.arrays || options.defaults)
                    object.updates = [];
                if (message.updates && message.updates.length) {
                    object.updates = [];
                    for (let j = 0; j < message.updates.length; ++j)
                        object.updates[j] = $root.lagrangian.testing.ModelForward.toObject(message.updates[j], options, q + 1);
                }
                return object;
            };

            /**
             * Converts this EnvUpdates to JSON.
             * @function toJSON
             * @memberof lagrangian.testing.EnvUpdates
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            EnvUpdates.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for EnvUpdates
             * @function getTypeUrl
             * @memberof lagrangian.testing.EnvUpdates
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            EnvUpdates.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/lagrangian.testing.EnvUpdates";
            };

            return EnvUpdates;
        })();

        testing.TaskUpdates = (function() {

            /**
             * Properties of a TaskUpdates.
             * @memberof lagrangian.testing
             * @interface ITaskUpdates
             * @property {number|Long|null} [taskIndex] TaskUpdates taskIndex
             * @property {Array.<lagrangian.testing.IEnvUpdates>|null} [environments] TaskUpdates environments
             */

            /**
             * Constructs a new TaskUpdates.
             * @memberof lagrangian.testing
             * @classdesc Represents a TaskUpdates.
             * @implements ITaskUpdates
             * @constructor
             * @param {lagrangian.testing.ITaskUpdates=} [properties] Properties to set
             */
            function TaskUpdates(properties) {
                this.environments = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null && keys[i] !== "__proto__")
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * TaskUpdates taskIndex.
             * @member {number|Long} taskIndex
             * @memberof lagrangian.testing.TaskUpdates
             * @instance
             */
            TaskUpdates.prototype.taskIndex = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * TaskUpdates environments.
             * @member {Array.<lagrangian.testing.IEnvUpdates>} environments
             * @memberof lagrangian.testing.TaskUpdates
             * @instance
             */
            TaskUpdates.prototype.environments = $util.emptyArray;

            /**
             * Creates a new TaskUpdates instance using the specified properties.
             * @function create
             * @memberof lagrangian.testing.TaskUpdates
             * @static
             * @param {lagrangian.testing.ITaskUpdates=} [properties] Properties to set
             * @returns {lagrangian.testing.TaskUpdates} TaskUpdates instance
             */
            TaskUpdates.create = function create(properties) {
                return new TaskUpdates(properties);
            };

            /**
             * Encodes the specified TaskUpdates message. Does not implicitly {@link lagrangian.testing.TaskUpdates.verify|verify} messages.
             * @function encode
             * @memberof lagrangian.testing.TaskUpdates
             * @static
             * @param {lagrangian.testing.ITaskUpdates} message TaskUpdates message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TaskUpdates.encode = function encode(message, writer, q) {
                if (!writer)
                    writer = $Writer.create();
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                if (message.taskIndex != null && Object.hasOwnProperty.call(message, "taskIndex"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int64(message.taskIndex);
                if (message.environments != null && message.environments.length)
                    for (let i = 0; i < message.environments.length; ++i)
                        $root.lagrangian.testing.EnvUpdates.encode(message.environments[i], writer.uint32(/* id 2, wireType 2 =*/18).fork(), q + 1).ldelim();
                return writer;
            };

            /**
             * Encodes the specified TaskUpdates message, length delimited. Does not implicitly {@link lagrangian.testing.TaskUpdates.verify|verify} messages.
             * @function encodeDelimited
             * @memberof lagrangian.testing.TaskUpdates
             * @static
             * @param {lagrangian.testing.ITaskUpdates} message TaskUpdates message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TaskUpdates.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer && writer.len ? writer.fork() : writer).ldelim();
            };

            /**
             * Decodes a TaskUpdates message from the specified reader or buffer.
             * @function decode
             * @memberof lagrangian.testing.TaskUpdates
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {lagrangian.testing.TaskUpdates} TaskUpdates
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TaskUpdates.decode = function decode(reader, length, error, long) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                if (long === undefined)
                    long = 0;
                if (long > $Reader.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.lagrangian.testing.TaskUpdates();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.taskIndex = reader.int64();
                            break;
                        }
                    case 2: {
                            if (!(message.environments && message.environments.length))
                                message.environments = [];
                            message.environments.push($root.lagrangian.testing.EnvUpdates.decode(reader, reader.uint32(), undefined, long + 1));
                            break;
                        }
                    default:
                        reader.skipType(tag & 7, long);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a TaskUpdates message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof lagrangian.testing.TaskUpdates
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {lagrangian.testing.TaskUpdates} TaskUpdates
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TaskUpdates.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a TaskUpdates message.
             * @function verify
             * @memberof lagrangian.testing.TaskUpdates
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            TaskUpdates.verify = function verify(message, long) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    return "maximum nesting depth exceeded";
                if (message.taskIndex != null && Object.hasOwnProperty.call(message, "taskIndex"))
                    if (!$util.isInteger(message.taskIndex) && !(message.taskIndex && $util.isInteger(message.taskIndex.low) && $util.isInteger(message.taskIndex.high)))
                        return "taskIndex: integer|Long expected";
                if (message.environments != null && Object.hasOwnProperty.call(message, "environments")) {
                    if (!Array.isArray(message.environments))
                        return "environments: array expected";
                    for (let i = 0; i < message.environments.length; ++i) {
                        let error = $root.lagrangian.testing.EnvUpdates.verify(message.environments[i], long + 1);
                        if (error)
                            return "environments." + error;
                    }
                }
                return null;
            };

            /**
             * Creates a TaskUpdates message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof lagrangian.testing.TaskUpdates
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {lagrangian.testing.TaskUpdates} TaskUpdates
             */
            TaskUpdates.fromObject = function fromObject(object, long) {
                if (object instanceof $root.lagrangian.testing.TaskUpdates)
                    return object;
                if (!$util.isObject(object))
                    throw TypeError(".lagrangian.testing.TaskUpdates: object expected");
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let message = new $root.lagrangian.testing.TaskUpdates();
                if (object.taskIndex != null)
                    if ($util.Long)
                        message.taskIndex = $util.Long.fromValue(object.taskIndex, false);
                    else if (typeof object.taskIndex === "string")
                        message.taskIndex = parseInt(object.taskIndex, 10);
                    else if (typeof object.taskIndex === "number")
                        message.taskIndex = object.taskIndex;
                    else if (typeof object.taskIndex === "object")
                        message.taskIndex = new $util.LongBits(object.taskIndex.low >>> 0, object.taskIndex.high >>> 0).toNumber();
                if (object.environments) {
                    if (!Array.isArray(object.environments))
                        throw TypeError(".lagrangian.testing.TaskUpdates.environments: array expected");
                    message.environments = [];
                    for (let i = 0; i < object.environments.length; ++i) {
                        if (!$util.isObject(object.environments[i]))
                            throw TypeError(".lagrangian.testing.TaskUpdates.environments: object expected");
                        message.environments[i] = $root.lagrangian.testing.EnvUpdates.fromObject(object.environments[i], long + 1);
                    }
                }
                return message;
            };

            /**
             * Creates a plain object from a TaskUpdates message. Also converts values to other types if specified.
             * @function toObject
             * @memberof lagrangian.testing.TaskUpdates
             * @static
             * @param {lagrangian.testing.TaskUpdates} message TaskUpdates
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            TaskUpdates.toObject = function toObject(message, options, q) {
                if (!options)
                    options = {};
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                let object = {};
                if (options.arrays || options.defaults)
                    object.environments = [];
                if (options.defaults)
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.taskIndex = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : typeof BigInt !== "undefined" && options.longs === BigInt ? long.toBigInt() : long;
                    } else
                        object.taskIndex = options.longs === String ? "0" : typeof BigInt !== "undefined" && options.longs === BigInt ? BigInt("0") : 0;
                if (message.taskIndex != null && Object.hasOwnProperty.call(message, "taskIndex"))
                    if (typeof BigInt !== "undefined" && options.longs === BigInt)
                        object.taskIndex = typeof message.taskIndex === "number" ? BigInt(message.taskIndex) : $util.Long.fromBits(message.taskIndex.low >>> 0, message.taskIndex.high >>> 0, false).toBigInt();
                    else if (typeof message.taskIndex === "number")
                        object.taskIndex = options.longs === String ? String(message.taskIndex) : message.taskIndex;
                    else
                        object.taskIndex = options.longs === String ? $util.Long.prototype.toString.call(message.taskIndex) : options.longs === Number ? new $util.LongBits(message.taskIndex.low >>> 0, message.taskIndex.high >>> 0).toNumber() : message.taskIndex;
                if (message.environments && message.environments.length) {
                    object.environments = [];
                    for (let j = 0; j < message.environments.length; ++j)
                        object.environments[j] = $root.lagrangian.testing.EnvUpdates.toObject(message.environments[j], options, q + 1);
                }
                return object;
            };

            /**
             * Converts this TaskUpdates to JSON.
             * @function toJSON
             * @memberof lagrangian.testing.TaskUpdates
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            TaskUpdates.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for TaskUpdates
             * @function getTypeUrl
             * @memberof lagrangian.testing.TaskUpdates
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            TaskUpdates.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/lagrangian.testing.TaskUpdates";
            };

            return TaskUpdates;
        })();

        testing.CameraFrames = (function() {

            /**
             * Properties of a CameraFrames.
             * @memberof lagrangian.testing
             * @interface ICameraFrames
             * @property {number|Long|null} [taskIndex] CameraFrames taskIndex
             * @property {number|Long|null} [nenv] CameraFrames nenv
             * @property {number|Long|null} [ncam] CameraFrames ncam
             * @property {number|Long|null} [t] CameraFrames t
             * @property {number|Long|null} [h] CameraFrames h
             * @property {number|Long|null} [w] CameraFrames w
             * @property {Uint8Array|null} [frames] CameraFrames frames
             */

            /**
             * Constructs a new CameraFrames.
             * @memberof lagrangian.testing
             * @classdesc Represents a CameraFrames.
             * @implements ICameraFrames
             * @constructor
             * @param {lagrangian.testing.ICameraFrames=} [properties] Properties to set
             */
            function CameraFrames(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null && keys[i] !== "__proto__")
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * CameraFrames taskIndex.
             * @member {number|Long} taskIndex
             * @memberof lagrangian.testing.CameraFrames
             * @instance
             */
            CameraFrames.prototype.taskIndex = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * CameraFrames nenv.
             * @member {number|Long} nenv
             * @memberof lagrangian.testing.CameraFrames
             * @instance
             */
            CameraFrames.prototype.nenv = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * CameraFrames ncam.
             * @member {number|Long} ncam
             * @memberof lagrangian.testing.CameraFrames
             * @instance
             */
            CameraFrames.prototype.ncam = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * CameraFrames t.
             * @member {number|Long} t
             * @memberof lagrangian.testing.CameraFrames
             * @instance
             */
            CameraFrames.prototype.t = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * CameraFrames h.
             * @member {number|Long} h
             * @memberof lagrangian.testing.CameraFrames
             * @instance
             */
            CameraFrames.prototype.h = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * CameraFrames w.
             * @member {number|Long} w
             * @memberof lagrangian.testing.CameraFrames
             * @instance
             */
            CameraFrames.prototype.w = $util.Long ? $util.Long.fromBits(0,0,false) : 0;

            /**
             * CameraFrames frames.
             * @member {Uint8Array} frames
             * @memberof lagrangian.testing.CameraFrames
             * @instance
             */
            CameraFrames.prototype.frames = $util.newBuffer([]);

            /**
             * Creates a new CameraFrames instance using the specified properties.
             * @function create
             * @memberof lagrangian.testing.CameraFrames
             * @static
             * @param {lagrangian.testing.ICameraFrames=} [properties] Properties to set
             * @returns {lagrangian.testing.CameraFrames} CameraFrames instance
             */
            CameraFrames.create = function create(properties) {
                return new CameraFrames(properties);
            };

            /**
             * Encodes the specified CameraFrames message. Does not implicitly {@link lagrangian.testing.CameraFrames.verify|verify} messages.
             * @function encode
             * @memberof lagrangian.testing.CameraFrames
             * @static
             * @param {lagrangian.testing.ICameraFrames} message CameraFrames message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            CameraFrames.encode = function encode(message, writer, q) {
                if (!writer)
                    writer = $Writer.create();
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                if (message.taskIndex != null && Object.hasOwnProperty.call(message, "taskIndex"))
                    writer.uint32(/* id 1, wireType 0 =*/8).int64(message.taskIndex);
                if (message.nenv != null && Object.hasOwnProperty.call(message, "nenv"))
                    writer.uint32(/* id 2, wireType 0 =*/16).int64(message.nenv);
                if (message.ncam != null && Object.hasOwnProperty.call(message, "ncam"))
                    writer.uint32(/* id 3, wireType 0 =*/24).int64(message.ncam);
                if (message.t != null && Object.hasOwnProperty.call(message, "t"))
                    writer.uint32(/* id 4, wireType 0 =*/32).int64(message.t);
                if (message.h != null && Object.hasOwnProperty.call(message, "h"))
                    writer.uint32(/* id 5, wireType 0 =*/40).int64(message.h);
                if (message.w != null && Object.hasOwnProperty.call(message, "w"))
                    writer.uint32(/* id 6, wireType 0 =*/48).int64(message.w);
                if (message.frames != null && Object.hasOwnProperty.call(message, "frames"))
                    writer.uint32(/* id 7, wireType 2 =*/58).bytes(message.frames);
                return writer;
            };

            /**
             * Encodes the specified CameraFrames message, length delimited. Does not implicitly {@link lagrangian.testing.CameraFrames.verify|verify} messages.
             * @function encodeDelimited
             * @memberof lagrangian.testing.CameraFrames
             * @static
             * @param {lagrangian.testing.ICameraFrames} message CameraFrames message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            CameraFrames.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer && writer.len ? writer.fork() : writer).ldelim();
            };

            /**
             * Decodes a CameraFrames message from the specified reader or buffer.
             * @function decode
             * @memberof lagrangian.testing.CameraFrames
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {lagrangian.testing.CameraFrames} CameraFrames
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            CameraFrames.decode = function decode(reader, length, error, long) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                if (long === undefined)
                    long = 0;
                if (long > $Reader.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.lagrangian.testing.CameraFrames();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.taskIndex = reader.int64();
                            break;
                        }
                    case 2: {
                            message.nenv = reader.int64();
                            break;
                        }
                    case 3: {
                            message.ncam = reader.int64();
                            break;
                        }
                    case 4: {
                            message.t = reader.int64();
                            break;
                        }
                    case 5: {
                            message.h = reader.int64();
                            break;
                        }
                    case 6: {
                            message.w = reader.int64();
                            break;
                        }
                    case 7: {
                            message.frames = reader.bytes();
                            break;
                        }
                    default:
                        reader.skipType(tag & 7, long);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a CameraFrames message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof lagrangian.testing.CameraFrames
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {lagrangian.testing.CameraFrames} CameraFrames
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            CameraFrames.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a CameraFrames message.
             * @function verify
             * @memberof lagrangian.testing.CameraFrames
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            CameraFrames.verify = function verify(message, long) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    return "maximum nesting depth exceeded";
                if (message.taskIndex != null && Object.hasOwnProperty.call(message, "taskIndex"))
                    if (!$util.isInteger(message.taskIndex) && !(message.taskIndex && $util.isInteger(message.taskIndex.low) && $util.isInteger(message.taskIndex.high)))
                        return "taskIndex: integer|Long expected";
                if (message.nenv != null && Object.hasOwnProperty.call(message, "nenv"))
                    if (!$util.isInteger(message.nenv) && !(message.nenv && $util.isInteger(message.nenv.low) && $util.isInteger(message.nenv.high)))
                        return "nenv: integer|Long expected";
                if (message.ncam != null && Object.hasOwnProperty.call(message, "ncam"))
                    if (!$util.isInteger(message.ncam) && !(message.ncam && $util.isInteger(message.ncam.low) && $util.isInteger(message.ncam.high)))
                        return "ncam: integer|Long expected";
                if (message.t != null && Object.hasOwnProperty.call(message, "t"))
                    if (!$util.isInteger(message.t) && !(message.t && $util.isInteger(message.t.low) && $util.isInteger(message.t.high)))
                        return "t: integer|Long expected";
                if (message.h != null && Object.hasOwnProperty.call(message, "h"))
                    if (!$util.isInteger(message.h) && !(message.h && $util.isInteger(message.h.low) && $util.isInteger(message.h.high)))
                        return "h: integer|Long expected";
                if (message.w != null && Object.hasOwnProperty.call(message, "w"))
                    if (!$util.isInteger(message.w) && !(message.w && $util.isInteger(message.w.low) && $util.isInteger(message.w.high)))
                        return "w: integer|Long expected";
                if (message.frames != null && Object.hasOwnProperty.call(message, "frames"))
                    if (!(message.frames && typeof message.frames.length === "number" || $util.isString(message.frames)))
                        return "frames: buffer expected";
                return null;
            };

            /**
             * Creates a CameraFrames message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof lagrangian.testing.CameraFrames
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {lagrangian.testing.CameraFrames} CameraFrames
             */
            CameraFrames.fromObject = function fromObject(object, long) {
                if (object instanceof $root.lagrangian.testing.CameraFrames)
                    return object;
                if (!$util.isObject(object))
                    throw TypeError(".lagrangian.testing.CameraFrames: object expected");
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let message = new $root.lagrangian.testing.CameraFrames();
                if (object.taskIndex != null)
                    if ($util.Long)
                        message.taskIndex = $util.Long.fromValue(object.taskIndex, false);
                    else if (typeof object.taskIndex === "string")
                        message.taskIndex = parseInt(object.taskIndex, 10);
                    else if (typeof object.taskIndex === "number")
                        message.taskIndex = object.taskIndex;
                    else if (typeof object.taskIndex === "object")
                        message.taskIndex = new $util.LongBits(object.taskIndex.low >>> 0, object.taskIndex.high >>> 0).toNumber();
                if (object.nenv != null)
                    if ($util.Long)
                        message.nenv = $util.Long.fromValue(object.nenv, false);
                    else if (typeof object.nenv === "string")
                        message.nenv = parseInt(object.nenv, 10);
                    else if (typeof object.nenv === "number")
                        message.nenv = object.nenv;
                    else if (typeof object.nenv === "object")
                        message.nenv = new $util.LongBits(object.nenv.low >>> 0, object.nenv.high >>> 0).toNumber();
                if (object.ncam != null)
                    if ($util.Long)
                        message.ncam = $util.Long.fromValue(object.ncam, false);
                    else if (typeof object.ncam === "string")
                        message.ncam = parseInt(object.ncam, 10);
                    else if (typeof object.ncam === "number")
                        message.ncam = object.ncam;
                    else if (typeof object.ncam === "object")
                        message.ncam = new $util.LongBits(object.ncam.low >>> 0, object.ncam.high >>> 0).toNumber();
                if (object.t != null)
                    if ($util.Long)
                        message.t = $util.Long.fromValue(object.t, false);
                    else if (typeof object.t === "string")
                        message.t = parseInt(object.t, 10);
                    else if (typeof object.t === "number")
                        message.t = object.t;
                    else if (typeof object.t === "object")
                        message.t = new $util.LongBits(object.t.low >>> 0, object.t.high >>> 0).toNumber();
                if (object.h != null)
                    if ($util.Long)
                        message.h = $util.Long.fromValue(object.h, false);
                    else if (typeof object.h === "string")
                        message.h = parseInt(object.h, 10);
                    else if (typeof object.h === "number")
                        message.h = object.h;
                    else if (typeof object.h === "object")
                        message.h = new $util.LongBits(object.h.low >>> 0, object.h.high >>> 0).toNumber();
                if (object.w != null)
                    if ($util.Long)
                        message.w = $util.Long.fromValue(object.w, false);
                    else if (typeof object.w === "string")
                        message.w = parseInt(object.w, 10);
                    else if (typeof object.w === "number")
                        message.w = object.w;
                    else if (typeof object.w === "object")
                        message.w = new $util.LongBits(object.w.low >>> 0, object.w.high >>> 0).toNumber();
                if (object.frames != null)
                    if (typeof object.frames === "string")
                        $util.base64.decode(object.frames, message.frames = $util.newBuffer($util.base64.length(object.frames)), 0);
                    else if (object.frames.length >= 0)
                        message.frames = object.frames;
                return message;
            };

            /**
             * Creates a plain object from a CameraFrames message. Also converts values to other types if specified.
             * @function toObject
             * @memberof lagrangian.testing.CameraFrames
             * @static
             * @param {lagrangian.testing.CameraFrames} message CameraFrames
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            CameraFrames.toObject = function toObject(message, options, q) {
                if (!options)
                    options = {};
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                let object = {};
                if (options.defaults) {
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.taskIndex = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : typeof BigInt !== "undefined" && options.longs === BigInt ? long.toBigInt() : long;
                    } else
                        object.taskIndex = options.longs === String ? "0" : typeof BigInt !== "undefined" && options.longs === BigInt ? BigInt("0") : 0;
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.nenv = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : typeof BigInt !== "undefined" && options.longs === BigInt ? long.toBigInt() : long;
                    } else
                        object.nenv = options.longs === String ? "0" : typeof BigInt !== "undefined" && options.longs === BigInt ? BigInt("0") : 0;
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.ncam = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : typeof BigInt !== "undefined" && options.longs === BigInt ? long.toBigInt() : long;
                    } else
                        object.ncam = options.longs === String ? "0" : typeof BigInt !== "undefined" && options.longs === BigInt ? BigInt("0") : 0;
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.t = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : typeof BigInt !== "undefined" && options.longs === BigInt ? long.toBigInt() : long;
                    } else
                        object.t = options.longs === String ? "0" : typeof BigInt !== "undefined" && options.longs === BigInt ? BigInt("0") : 0;
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.h = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : typeof BigInt !== "undefined" && options.longs === BigInt ? long.toBigInt() : long;
                    } else
                        object.h = options.longs === String ? "0" : typeof BigInt !== "undefined" && options.longs === BigInt ? BigInt("0") : 0;
                    if ($util.Long) {
                        let long = new $util.Long(0, 0, false);
                        object.w = options.longs === String ? long.toString() : options.longs === Number ? long.toNumber() : typeof BigInt !== "undefined" && options.longs === BigInt ? long.toBigInt() : long;
                    } else
                        object.w = options.longs === String ? "0" : typeof BigInt !== "undefined" && options.longs === BigInt ? BigInt("0") : 0;
                    if (options.bytes === String)
                        object.frames = "";
                    else {
                        object.frames = [];
                        if (options.bytes !== Array)
                            object.frames = $util.newBuffer(object.frames);
                    }
                }
                if (message.taskIndex != null && Object.hasOwnProperty.call(message, "taskIndex"))
                    if (typeof BigInt !== "undefined" && options.longs === BigInt)
                        object.taskIndex = typeof message.taskIndex === "number" ? BigInt(message.taskIndex) : $util.Long.fromBits(message.taskIndex.low >>> 0, message.taskIndex.high >>> 0, false).toBigInt();
                    else if (typeof message.taskIndex === "number")
                        object.taskIndex = options.longs === String ? String(message.taskIndex) : message.taskIndex;
                    else
                        object.taskIndex = options.longs === String ? $util.Long.prototype.toString.call(message.taskIndex) : options.longs === Number ? new $util.LongBits(message.taskIndex.low >>> 0, message.taskIndex.high >>> 0).toNumber() : message.taskIndex;
                if (message.nenv != null && Object.hasOwnProperty.call(message, "nenv"))
                    if (typeof BigInt !== "undefined" && options.longs === BigInt)
                        object.nenv = typeof message.nenv === "number" ? BigInt(message.nenv) : $util.Long.fromBits(message.nenv.low >>> 0, message.nenv.high >>> 0, false).toBigInt();
                    else if (typeof message.nenv === "number")
                        object.nenv = options.longs === String ? String(message.nenv) : message.nenv;
                    else
                        object.nenv = options.longs === String ? $util.Long.prototype.toString.call(message.nenv) : options.longs === Number ? new $util.LongBits(message.nenv.low >>> 0, message.nenv.high >>> 0).toNumber() : message.nenv;
                if (message.ncam != null && Object.hasOwnProperty.call(message, "ncam"))
                    if (typeof BigInt !== "undefined" && options.longs === BigInt)
                        object.ncam = typeof message.ncam === "number" ? BigInt(message.ncam) : $util.Long.fromBits(message.ncam.low >>> 0, message.ncam.high >>> 0, false).toBigInt();
                    else if (typeof message.ncam === "number")
                        object.ncam = options.longs === String ? String(message.ncam) : message.ncam;
                    else
                        object.ncam = options.longs === String ? $util.Long.prototype.toString.call(message.ncam) : options.longs === Number ? new $util.LongBits(message.ncam.low >>> 0, message.ncam.high >>> 0).toNumber() : message.ncam;
                if (message.t != null && Object.hasOwnProperty.call(message, "t"))
                    if (typeof BigInt !== "undefined" && options.longs === BigInt)
                        object.t = typeof message.t === "number" ? BigInt(message.t) : $util.Long.fromBits(message.t.low >>> 0, message.t.high >>> 0, false).toBigInt();
                    else if (typeof message.t === "number")
                        object.t = options.longs === String ? String(message.t) : message.t;
                    else
                        object.t = options.longs === String ? $util.Long.prototype.toString.call(message.t) : options.longs === Number ? new $util.LongBits(message.t.low >>> 0, message.t.high >>> 0).toNumber() : message.t;
                if (message.h != null && Object.hasOwnProperty.call(message, "h"))
                    if (typeof BigInt !== "undefined" && options.longs === BigInt)
                        object.h = typeof message.h === "number" ? BigInt(message.h) : $util.Long.fromBits(message.h.low >>> 0, message.h.high >>> 0, false).toBigInt();
                    else if (typeof message.h === "number")
                        object.h = options.longs === String ? String(message.h) : message.h;
                    else
                        object.h = options.longs === String ? $util.Long.prototype.toString.call(message.h) : options.longs === Number ? new $util.LongBits(message.h.low >>> 0, message.h.high >>> 0).toNumber() : message.h;
                if (message.w != null && Object.hasOwnProperty.call(message, "w"))
                    if (typeof BigInt !== "undefined" && options.longs === BigInt)
                        object.w = typeof message.w === "number" ? BigInt(message.w) : $util.Long.fromBits(message.w.low >>> 0, message.w.high >>> 0, false).toBigInt();
                    else if (typeof message.w === "number")
                        object.w = options.longs === String ? String(message.w) : message.w;
                    else
                        object.w = options.longs === String ? $util.Long.prototype.toString.call(message.w) : options.longs === Number ? new $util.LongBits(message.w.low >>> 0, message.w.high >>> 0).toNumber() : message.w;
                if (message.frames != null && Object.hasOwnProperty.call(message, "frames"))
                    object.frames = options.bytes === String ? $util.base64.encode(message.frames, 0, message.frames.length) : options.bytes === Array ? Array.prototype.slice.call(message.frames) : message.frames;
                return object;
            };

            /**
             * Converts this CameraFrames to JSON.
             * @function toJSON
             * @memberof lagrangian.testing.CameraFrames
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            CameraFrames.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for CameraFrames
             * @function getTypeUrl
             * @memberof lagrangian.testing.CameraFrames
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            CameraFrames.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/lagrangian.testing.CameraFrames";
            };

            return CameraFrames;
        })();

        testing.TestUpdate = (function() {

            /**
             * Properties of a TestUpdate.
             * @memberof lagrangian.testing
             * @interface ITestUpdate
             * @property {string|null} [id] TestUpdate id
             * @property {Array.<lagrangian.testing.ITaskUpdates>|null} [simulationState] TestUpdate simulationState
             * @property {Array.<lagrangian.testing.ICameraFrames>|null} [frames] TestUpdate frames
             */

            /**
             * Constructs a new TestUpdate.
             * @memberof lagrangian.testing
             * @classdesc Represents a TestUpdate.
             * @implements ITestUpdate
             * @constructor
             * @param {lagrangian.testing.ITestUpdate=} [properties] Properties to set
             */
            function TestUpdate(properties) {
                this.simulationState = [];
                this.frames = [];
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null && keys[i] !== "__proto__")
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * TestUpdate id.
             * @member {string} id
             * @memberof lagrangian.testing.TestUpdate
             * @instance
             */
            TestUpdate.prototype.id = "";

            /**
             * TestUpdate simulationState.
             * @member {Array.<lagrangian.testing.ITaskUpdates>} simulationState
             * @memberof lagrangian.testing.TestUpdate
             * @instance
             */
            TestUpdate.prototype.simulationState = $util.emptyArray;

            /**
             * TestUpdate frames.
             * @member {Array.<lagrangian.testing.ICameraFrames>} frames
             * @memberof lagrangian.testing.TestUpdate
             * @instance
             */
            TestUpdate.prototype.frames = $util.emptyArray;

            /**
             * Creates a new TestUpdate instance using the specified properties.
             * @function create
             * @memberof lagrangian.testing.TestUpdate
             * @static
             * @param {lagrangian.testing.ITestUpdate=} [properties] Properties to set
             * @returns {lagrangian.testing.TestUpdate} TestUpdate instance
             */
            TestUpdate.create = function create(properties) {
                return new TestUpdate(properties);
            };

            /**
             * Encodes the specified TestUpdate message. Does not implicitly {@link lagrangian.testing.TestUpdate.verify|verify} messages.
             * @function encode
             * @memberof lagrangian.testing.TestUpdate
             * @static
             * @param {lagrangian.testing.ITestUpdate} message TestUpdate message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TestUpdate.encode = function encode(message, writer, q) {
                if (!writer)
                    writer = $Writer.create();
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                    writer.uint32(/* id 1, wireType 2 =*/10).string(message.id);
                if (message.simulationState != null && message.simulationState.length)
                    for (let i = 0; i < message.simulationState.length; ++i)
                        $root.lagrangian.testing.TaskUpdates.encode(message.simulationState[i], writer.uint32(/* id 2, wireType 2 =*/18).fork(), q + 1).ldelim();
                if (message.frames != null && message.frames.length)
                    for (let i = 0; i < message.frames.length; ++i)
                        $root.lagrangian.testing.CameraFrames.encode(message.frames[i], writer.uint32(/* id 3, wireType 2 =*/26).fork(), q + 1).ldelim();
                return writer;
            };

            /**
             * Encodes the specified TestUpdate message, length delimited. Does not implicitly {@link lagrangian.testing.TestUpdate.verify|verify} messages.
             * @function encodeDelimited
             * @memberof lagrangian.testing.TestUpdate
             * @static
             * @param {lagrangian.testing.ITestUpdate} message TestUpdate message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TestUpdate.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer && writer.len ? writer.fork() : writer).ldelim();
            };

            /**
             * Decodes a TestUpdate message from the specified reader or buffer.
             * @function decode
             * @memberof lagrangian.testing.TestUpdate
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {lagrangian.testing.TestUpdate} TestUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TestUpdate.decode = function decode(reader, length, error, long) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                if (long === undefined)
                    long = 0;
                if (long > $Reader.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.lagrangian.testing.TestUpdate();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.id = reader.string();
                            break;
                        }
                    case 2: {
                            if (!(message.simulationState && message.simulationState.length))
                                message.simulationState = [];
                            message.simulationState.push($root.lagrangian.testing.TaskUpdates.decode(reader, reader.uint32(), undefined, long + 1));
                            break;
                        }
                    case 3: {
                            if (!(message.frames && message.frames.length))
                                message.frames = [];
                            message.frames.push($root.lagrangian.testing.CameraFrames.decode(reader, reader.uint32(), undefined, long + 1));
                            break;
                        }
                    default:
                        reader.skipType(tag & 7, long);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a TestUpdate message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof lagrangian.testing.TestUpdate
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {lagrangian.testing.TestUpdate} TestUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TestUpdate.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a TestUpdate message.
             * @function verify
             * @memberof lagrangian.testing.TestUpdate
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            TestUpdate.verify = function verify(message, long) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    return "maximum nesting depth exceeded";
                if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                    if (!$util.isString(message.id))
                        return "id: string expected";
                if (message.simulationState != null && Object.hasOwnProperty.call(message, "simulationState")) {
                    if (!Array.isArray(message.simulationState))
                        return "simulationState: array expected";
                    for (let i = 0; i < message.simulationState.length; ++i) {
                        let error = $root.lagrangian.testing.TaskUpdates.verify(message.simulationState[i], long + 1);
                        if (error)
                            return "simulationState." + error;
                    }
                }
                if (message.frames != null && Object.hasOwnProperty.call(message, "frames")) {
                    if (!Array.isArray(message.frames))
                        return "frames: array expected";
                    for (let i = 0; i < message.frames.length; ++i) {
                        let error = $root.lagrangian.testing.CameraFrames.verify(message.frames[i], long + 1);
                        if (error)
                            return "frames." + error;
                    }
                }
                return null;
            };

            /**
             * Creates a TestUpdate message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof lagrangian.testing.TestUpdate
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {lagrangian.testing.TestUpdate} TestUpdate
             */
            TestUpdate.fromObject = function fromObject(object, long) {
                if (object instanceof $root.lagrangian.testing.TestUpdate)
                    return object;
                if (!$util.isObject(object))
                    throw TypeError(".lagrangian.testing.TestUpdate: object expected");
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let message = new $root.lagrangian.testing.TestUpdate();
                if (object.id != null)
                    message.id = String(object.id);
                if (object.simulationState) {
                    if (!Array.isArray(object.simulationState))
                        throw TypeError(".lagrangian.testing.TestUpdate.simulationState: array expected");
                    message.simulationState = [];
                    for (let i = 0; i < object.simulationState.length; ++i) {
                        if (!$util.isObject(object.simulationState[i]))
                            throw TypeError(".lagrangian.testing.TestUpdate.simulationState: object expected");
                        message.simulationState[i] = $root.lagrangian.testing.TaskUpdates.fromObject(object.simulationState[i], long + 1);
                    }
                }
                if (object.frames) {
                    if (!Array.isArray(object.frames))
                        throw TypeError(".lagrangian.testing.TestUpdate.frames: array expected");
                    message.frames = [];
                    for (let i = 0; i < object.frames.length; ++i) {
                        if (!$util.isObject(object.frames[i]))
                            throw TypeError(".lagrangian.testing.TestUpdate.frames: object expected");
                        message.frames[i] = $root.lagrangian.testing.CameraFrames.fromObject(object.frames[i], long + 1);
                    }
                }
                return message;
            };

            /**
             * Creates a plain object from a TestUpdate message. Also converts values to other types if specified.
             * @function toObject
             * @memberof lagrangian.testing.TestUpdate
             * @static
             * @param {lagrangian.testing.TestUpdate} message TestUpdate
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            TestUpdate.toObject = function toObject(message, options, q) {
                if (!options)
                    options = {};
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                let object = {};
                if (options.arrays || options.defaults) {
                    object.simulationState = [];
                    object.frames = [];
                }
                if (options.defaults)
                    object.id = "";
                if (message.id != null && Object.hasOwnProperty.call(message, "id"))
                    object.id = message.id;
                if (message.simulationState && message.simulationState.length) {
                    object.simulationState = [];
                    for (let j = 0; j < message.simulationState.length; ++j)
                        object.simulationState[j] = $root.lagrangian.testing.TaskUpdates.toObject(message.simulationState[j], options, q + 1);
                }
                if (message.frames && message.frames.length) {
                    object.frames = [];
                    for (let j = 0; j < message.frames.length; ++j)
                        object.frames[j] = $root.lagrangian.testing.CameraFrames.toObject(message.frames[j], options, q + 1);
                }
                return object;
            };

            /**
             * Converts this TestUpdate to JSON.
             * @function toJSON
             * @memberof lagrangian.testing.TestUpdate
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            TestUpdate.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for TestUpdate
             * @function getTypeUrl
             * @memberof lagrangian.testing.TestUpdate
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            TestUpdate.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/lagrangian.testing.TestUpdate";
            };

            return TestUpdate;
        })();

        testing.TestingStreamMessage = (function() {

            /**
             * Properties of a TestingStreamMessage.
             * @memberof lagrangian.testing
             * @interface ITestingStreamMessage
             * @property {lagrangian.testing.ITestUpdate|null} [testUpdate] TestingStreamMessage testUpdate
             */

            /**
             * Constructs a new TestingStreamMessage.
             * @memberof lagrangian.testing
             * @classdesc Represents a TestingStreamMessage.
             * @implements ITestingStreamMessage
             * @constructor
             * @param {lagrangian.testing.ITestingStreamMessage=} [properties] Properties to set
             */
            function TestingStreamMessage(properties) {
                if (properties)
                    for (let keys = Object.keys(properties), i = 0; i < keys.length; ++i)
                        if (properties[keys[i]] != null && keys[i] !== "__proto__")
                            this[keys[i]] = properties[keys[i]];
            }

            /**
             * TestingStreamMessage testUpdate.
             * @member {lagrangian.testing.ITestUpdate|null|undefined} testUpdate
             * @memberof lagrangian.testing.TestingStreamMessage
             * @instance
             */
            TestingStreamMessage.prototype.testUpdate = null;

            // OneOf field names bound to virtual getters and setters
            let $oneOfFields;

            /**
             * TestingStreamMessage message.
             * @member {"testUpdate"|undefined} message
             * @memberof lagrangian.testing.TestingStreamMessage
             * @instance
             */
            Object.defineProperty(TestingStreamMessage.prototype, "message", {
                get: $util.oneOfGetter($oneOfFields = ["testUpdate"]),
                set: $util.oneOfSetter($oneOfFields)
            });

            /**
             * Creates a new TestingStreamMessage instance using the specified properties.
             * @function create
             * @memberof lagrangian.testing.TestingStreamMessage
             * @static
             * @param {lagrangian.testing.ITestingStreamMessage=} [properties] Properties to set
             * @returns {lagrangian.testing.TestingStreamMessage} TestingStreamMessage instance
             */
            TestingStreamMessage.create = function create(properties) {
                return new TestingStreamMessage(properties);
            };

            /**
             * Encodes the specified TestingStreamMessage message. Does not implicitly {@link lagrangian.testing.TestingStreamMessage.verify|verify} messages.
             * @function encode
             * @memberof lagrangian.testing.TestingStreamMessage
             * @static
             * @param {lagrangian.testing.ITestingStreamMessage} message TestingStreamMessage message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TestingStreamMessage.encode = function encode(message, writer, q) {
                if (!writer)
                    writer = $Writer.create();
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                if (message.testUpdate != null && Object.hasOwnProperty.call(message, "testUpdate"))
                    $root.lagrangian.testing.TestUpdate.encode(message.testUpdate, writer.uint32(/* id 1, wireType 2 =*/10).fork(), q + 1).ldelim();
                return writer;
            };

            /**
             * Encodes the specified TestingStreamMessage message, length delimited. Does not implicitly {@link lagrangian.testing.TestingStreamMessage.verify|verify} messages.
             * @function encodeDelimited
             * @memberof lagrangian.testing.TestingStreamMessage
             * @static
             * @param {lagrangian.testing.ITestingStreamMessage} message TestingStreamMessage message or plain object to encode
             * @param {$protobuf.Writer} [writer] Writer to encode to
             * @returns {$protobuf.Writer} Writer
             */
            TestingStreamMessage.encodeDelimited = function encodeDelimited(message, writer) {
                return this.encode(message, writer && writer.len ? writer.fork() : writer).ldelim();
            };

            /**
             * Decodes a TestingStreamMessage message from the specified reader or buffer.
             * @function decode
             * @memberof lagrangian.testing.TestingStreamMessage
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @param {number} [length] Message length if known beforehand
             * @returns {lagrangian.testing.TestingStreamMessage} TestingStreamMessage
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TestingStreamMessage.decode = function decode(reader, length, error, long) {
                if (!(reader instanceof $Reader))
                    reader = $Reader.create(reader);
                if (long === undefined)
                    long = 0;
                if (long > $Reader.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let end = length === undefined ? reader.len : reader.pos + length, message = new $root.lagrangian.testing.TestingStreamMessage();
                while (reader.pos < end) {
                    let tag = reader.uint32();
                    if (tag === error)
                        break;
                    switch (tag >>> 3) {
                    case 1: {
                            message.testUpdate = $root.lagrangian.testing.TestUpdate.decode(reader, reader.uint32(), undefined, long + 1);
                            break;
                        }
                    default:
                        reader.skipType(tag & 7, long);
                        break;
                    }
                }
                return message;
            };

            /**
             * Decodes a TestingStreamMessage message from the specified reader or buffer, length delimited.
             * @function decodeDelimited
             * @memberof lagrangian.testing.TestingStreamMessage
             * @static
             * @param {$protobuf.Reader|Uint8Array} reader Reader or buffer to decode from
             * @returns {lagrangian.testing.TestingStreamMessage} TestingStreamMessage
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            TestingStreamMessage.decodeDelimited = function decodeDelimited(reader) {
                if (!(reader instanceof $Reader))
                    reader = new $Reader(reader);
                return this.decode(reader, reader.uint32());
            };

            /**
             * Verifies a TestingStreamMessage message.
             * @function verify
             * @memberof lagrangian.testing.TestingStreamMessage
             * @static
             * @param {Object.<string,*>} message Plain object to verify
             * @returns {string|null} `null` if valid, otherwise the reason why it is not
             */
            TestingStreamMessage.verify = function verify(message, long) {
                if (typeof message !== "object" || message === null)
                    return "object expected";
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    return "maximum nesting depth exceeded";
                let properties = {};
                if (message.testUpdate != null && Object.hasOwnProperty.call(message, "testUpdate")) {
                    properties.message = 1;
                    {
                        let error = $root.lagrangian.testing.TestUpdate.verify(message.testUpdate, long + 1);
                        if (error)
                            return "testUpdate." + error;
                    }
                }
                return null;
            };

            /**
             * Creates a TestingStreamMessage message from a plain object. Also converts values to their respective internal types.
             * @function fromObject
             * @memberof lagrangian.testing.TestingStreamMessage
             * @static
             * @param {Object.<string,*>} object Plain object
             * @returns {lagrangian.testing.TestingStreamMessage} TestingStreamMessage
             */
            TestingStreamMessage.fromObject = function fromObject(object, long) {
                if (object instanceof $root.lagrangian.testing.TestingStreamMessage)
                    return object;
                if (!$util.isObject(object))
                    throw TypeError(".lagrangian.testing.TestingStreamMessage: object expected");
                if (long === undefined)
                    long = 0;
                if (long > $util.recursionLimit)
                    throw Error("maximum nesting depth exceeded");
                let message = new $root.lagrangian.testing.TestingStreamMessage();
                if (object.testUpdate != null) {
                    if (!$util.isObject(object.testUpdate))
                        throw TypeError(".lagrangian.testing.TestingStreamMessage.testUpdate: object expected");
                    message.testUpdate = $root.lagrangian.testing.TestUpdate.fromObject(object.testUpdate, long + 1);
                }
                return message;
            };

            /**
             * Creates a plain object from a TestingStreamMessage message. Also converts values to other types if specified.
             * @function toObject
             * @memberof lagrangian.testing.TestingStreamMessage
             * @static
             * @param {lagrangian.testing.TestingStreamMessage} message TestingStreamMessage
             * @param {$protobuf.IConversionOptions} [options] Conversion options
             * @returns {Object.<string,*>} Plain object
             */
            TestingStreamMessage.toObject = function toObject(message, options, q) {
                if (!options)
                    options = {};
                if (q === undefined)
                    q = 0;
                if (q > $util.recursionLimit)
                    throw Error("max depth exceeded");
                let object = {};
                if (message.testUpdate != null && Object.hasOwnProperty.call(message, "testUpdate")) {
                    object.testUpdate = $root.lagrangian.testing.TestUpdate.toObject(message.testUpdate, options, q + 1);
                    if (options.oneofs)
                        object.message = "testUpdate";
                }
                return object;
            };

            /**
             * Converts this TestingStreamMessage to JSON.
             * @function toJSON
             * @memberof lagrangian.testing.TestingStreamMessage
             * @instance
             * @returns {Object.<string,*>} JSON object
             */
            TestingStreamMessage.prototype.toJSON = function toJSON() {
                return this.constructor.toObject(this, $protobuf.util.toJSONOptions);
            };

            /**
             * Gets the default type url for TestingStreamMessage
             * @function getTypeUrl
             * @memberof lagrangian.testing.TestingStreamMessage
             * @static
             * @param {string} [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns {string} The default type url
             */
            TestingStreamMessage.getTypeUrl = function getTypeUrl(typeUrlPrefix) {
                if (typeUrlPrefix === undefined) {
                    typeUrlPrefix = "type.googleapis.com";
                }
                return typeUrlPrefix + "/lagrangian.testing.TestingStreamMessage";
            };

            return TestingStreamMessage;
        })();

        return testing;
    })();

    return lagrangian;
})();

export { $root as default };
