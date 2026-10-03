import * as $protobuf from "protobufjs";
import Long = require("long");
/** Namespace lagrangian. */
export namespace lagrangian {

    /** Namespace testing. */
    namespace testing {

        /** Properties of a ModelForward. */
        interface IModelForward {

            /** ModelForward nbody */
            nbody?: (number|Long|null);

            /** ModelForward xpos */
            xpos?: (number[]|null);

            /** ModelForward xquat */
            xquat?: (number[]|null);
        }

        /** Represents a ModelForward. */
        class ModelForward implements IModelForward {

            /**
             * Constructs a new ModelForward.
             * @param [properties] Properties to set
             */
            constructor(properties?: lagrangian.testing.IModelForward);

            /** ModelForward nbody. */
            public nbody: (number|Long);

            /** ModelForward xpos. */
            public xpos: number[];

            /** ModelForward xquat. */
            public xquat: number[];

            /**
             * Creates a new ModelForward instance using the specified properties.
             * @param [properties] Properties to set
             * @returns ModelForward instance
             */
            public static create(properties?: lagrangian.testing.IModelForward): lagrangian.testing.ModelForward;

            /**
             * Encodes the specified ModelForward message. Does not implicitly {@link lagrangian.testing.ModelForward.verify|verify} messages.
             * @param message ModelForward message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: lagrangian.testing.IModelForward, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified ModelForward message, length delimited. Does not implicitly {@link lagrangian.testing.ModelForward.verify|verify} messages.
             * @param message ModelForward message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: lagrangian.testing.IModelForward, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a ModelForward message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns ModelForward
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): lagrangian.testing.ModelForward;

            /**
             * Decodes a ModelForward message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns ModelForward
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): lagrangian.testing.ModelForward;

            /**
             * Verifies a ModelForward message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a ModelForward message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns ModelForward
             */
            public static fromObject(object: { [k: string]: any }): lagrangian.testing.ModelForward;

            /**
             * Creates a plain object from a ModelForward message. Also converts values to other types if specified.
             * @param message ModelForward
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: lagrangian.testing.ModelForward, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this ModelForward to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for ModelForward
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of an EnvUpdates. */
        interface IEnvUpdates {

            /** EnvUpdates updates */
            updates?: (lagrangian.testing.IModelForward[]|null);
        }

        /** Represents an EnvUpdates. */
        class EnvUpdates implements IEnvUpdates {

            /**
             * Constructs a new EnvUpdates.
             * @param [properties] Properties to set
             */
            constructor(properties?: lagrangian.testing.IEnvUpdates);

            /** EnvUpdates updates. */
            public updates: lagrangian.testing.IModelForward[];

            /**
             * Creates a new EnvUpdates instance using the specified properties.
             * @param [properties] Properties to set
             * @returns EnvUpdates instance
             */
            public static create(properties?: lagrangian.testing.IEnvUpdates): lagrangian.testing.EnvUpdates;

            /**
             * Encodes the specified EnvUpdates message. Does not implicitly {@link lagrangian.testing.EnvUpdates.verify|verify} messages.
             * @param message EnvUpdates message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: lagrangian.testing.IEnvUpdates, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified EnvUpdates message, length delimited. Does not implicitly {@link lagrangian.testing.EnvUpdates.verify|verify} messages.
             * @param message EnvUpdates message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: lagrangian.testing.IEnvUpdates, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes an EnvUpdates message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns EnvUpdates
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): lagrangian.testing.EnvUpdates;

            /**
             * Decodes an EnvUpdates message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns EnvUpdates
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): lagrangian.testing.EnvUpdates;

            /**
             * Verifies an EnvUpdates message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates an EnvUpdates message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns EnvUpdates
             */
            public static fromObject(object: { [k: string]: any }): lagrangian.testing.EnvUpdates;

            /**
             * Creates a plain object from an EnvUpdates message. Also converts values to other types if specified.
             * @param message EnvUpdates
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: lagrangian.testing.EnvUpdates, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this EnvUpdates to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for EnvUpdates
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a TaskUpdates. */
        interface ITaskUpdates {

            /** TaskUpdates taskIndex */
            taskIndex?: (number|Long|null);

            /** TaskUpdates environments */
            environments?: (lagrangian.testing.IEnvUpdates[]|null);
        }

        /** Represents a TaskUpdates. */
        class TaskUpdates implements ITaskUpdates {

            /**
             * Constructs a new TaskUpdates.
             * @param [properties] Properties to set
             */
            constructor(properties?: lagrangian.testing.ITaskUpdates);

            /** TaskUpdates taskIndex. */
            public taskIndex: (number|Long);

            /** TaskUpdates environments. */
            public environments: lagrangian.testing.IEnvUpdates[];

            /**
             * Creates a new TaskUpdates instance using the specified properties.
             * @param [properties] Properties to set
             * @returns TaskUpdates instance
             */
            public static create(properties?: lagrangian.testing.ITaskUpdates): lagrangian.testing.TaskUpdates;

            /**
             * Encodes the specified TaskUpdates message. Does not implicitly {@link lagrangian.testing.TaskUpdates.verify|verify} messages.
             * @param message TaskUpdates message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: lagrangian.testing.ITaskUpdates, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified TaskUpdates message, length delimited. Does not implicitly {@link lagrangian.testing.TaskUpdates.verify|verify} messages.
             * @param message TaskUpdates message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: lagrangian.testing.ITaskUpdates, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a TaskUpdates message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns TaskUpdates
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): lagrangian.testing.TaskUpdates;

            /**
             * Decodes a TaskUpdates message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns TaskUpdates
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): lagrangian.testing.TaskUpdates;

            /**
             * Verifies a TaskUpdates message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a TaskUpdates message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns TaskUpdates
             */
            public static fromObject(object: { [k: string]: any }): lagrangian.testing.TaskUpdates;

            /**
             * Creates a plain object from a TaskUpdates message. Also converts values to other types if specified.
             * @param message TaskUpdates
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: lagrangian.testing.TaskUpdates, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this TaskUpdates to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for TaskUpdates
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a CameraFrames. */
        interface ICameraFrames {

            /** CameraFrames taskIndex */
            taskIndex?: (number|Long|null);

            /** CameraFrames nenv */
            nenv?: (number|Long|null);

            /** CameraFrames ncam */
            ncam?: (number|Long|null);

            /** CameraFrames t */
            t?: (number|Long|null);

            /** CameraFrames h */
            h?: (number|Long|null);

            /** CameraFrames w */
            w?: (number|Long|null);

            /** CameraFrames frames */
            frames?: (Uint8Array|null);
        }

        /** Represents a CameraFrames. */
        class CameraFrames implements ICameraFrames {

            /**
             * Constructs a new CameraFrames.
             * @param [properties] Properties to set
             */
            constructor(properties?: lagrangian.testing.ICameraFrames);

            /** CameraFrames taskIndex. */
            public taskIndex: (number|Long);

            /** CameraFrames nenv. */
            public nenv: (number|Long);

            /** CameraFrames ncam. */
            public ncam: (number|Long);

            /** CameraFrames t. */
            public t: (number|Long);

            /** CameraFrames h. */
            public h: (number|Long);

            /** CameraFrames w. */
            public w: (number|Long);

            /** CameraFrames frames. */
            public frames: Uint8Array;

            /**
             * Creates a new CameraFrames instance using the specified properties.
             * @param [properties] Properties to set
             * @returns CameraFrames instance
             */
            public static create(properties?: lagrangian.testing.ICameraFrames): lagrangian.testing.CameraFrames;

            /**
             * Encodes the specified CameraFrames message. Does not implicitly {@link lagrangian.testing.CameraFrames.verify|verify} messages.
             * @param message CameraFrames message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: lagrangian.testing.ICameraFrames, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified CameraFrames message, length delimited. Does not implicitly {@link lagrangian.testing.CameraFrames.verify|verify} messages.
             * @param message CameraFrames message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: lagrangian.testing.ICameraFrames, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a CameraFrames message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns CameraFrames
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): lagrangian.testing.CameraFrames;

            /**
             * Decodes a CameraFrames message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns CameraFrames
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): lagrangian.testing.CameraFrames;

            /**
             * Verifies a CameraFrames message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a CameraFrames message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns CameraFrames
             */
            public static fromObject(object: { [k: string]: any }): lagrangian.testing.CameraFrames;

            /**
             * Creates a plain object from a CameraFrames message. Also converts values to other types if specified.
             * @param message CameraFrames
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: lagrangian.testing.CameraFrames, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this CameraFrames to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for CameraFrames
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a TestUpdate. */
        interface ITestUpdate {

            /** TestUpdate id */
            id?: (string|null);

            /** TestUpdate simulationState */
            simulationState?: (lagrangian.testing.ITaskUpdates[]|null);

            /** TestUpdate frames */
            frames?: (lagrangian.testing.ICameraFrames[]|null);
        }

        /** Represents a TestUpdate. */
        class TestUpdate implements ITestUpdate {

            /**
             * Constructs a new TestUpdate.
             * @param [properties] Properties to set
             */
            constructor(properties?: lagrangian.testing.ITestUpdate);

            /** TestUpdate id. */
            public id: string;

            /** TestUpdate simulationState. */
            public simulationState: lagrangian.testing.ITaskUpdates[];

            /** TestUpdate frames. */
            public frames: lagrangian.testing.ICameraFrames[];

            /**
             * Creates a new TestUpdate instance using the specified properties.
             * @param [properties] Properties to set
             * @returns TestUpdate instance
             */
            public static create(properties?: lagrangian.testing.ITestUpdate): lagrangian.testing.TestUpdate;

            /**
             * Encodes the specified TestUpdate message. Does not implicitly {@link lagrangian.testing.TestUpdate.verify|verify} messages.
             * @param message TestUpdate message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: lagrangian.testing.ITestUpdate, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified TestUpdate message, length delimited. Does not implicitly {@link lagrangian.testing.TestUpdate.verify|verify} messages.
             * @param message TestUpdate message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: lagrangian.testing.ITestUpdate, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a TestUpdate message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns TestUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): lagrangian.testing.TestUpdate;

            /**
             * Decodes a TestUpdate message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns TestUpdate
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): lagrangian.testing.TestUpdate;

            /**
             * Verifies a TestUpdate message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a TestUpdate message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns TestUpdate
             */
            public static fromObject(object: { [k: string]: any }): lagrangian.testing.TestUpdate;

            /**
             * Creates a plain object from a TestUpdate message. Also converts values to other types if specified.
             * @param message TestUpdate
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: lagrangian.testing.TestUpdate, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this TestUpdate to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for TestUpdate
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }

        /** Properties of a TestingStreamMessage. */
        interface ITestingStreamMessage {

            /** TestingStreamMessage testUpdate */
            testUpdate?: (lagrangian.testing.ITestUpdate|null);
        }

        /** Represents a TestingStreamMessage. */
        class TestingStreamMessage implements ITestingStreamMessage {

            /**
             * Constructs a new TestingStreamMessage.
             * @param [properties] Properties to set
             */
            constructor(properties?: lagrangian.testing.ITestingStreamMessage);

            /** TestingStreamMessage testUpdate. */
            public testUpdate?: (lagrangian.testing.ITestUpdate|null);

            /** TestingStreamMessage message. */
            public message?: "testUpdate";

            /**
             * Creates a new TestingStreamMessage instance using the specified properties.
             * @param [properties] Properties to set
             * @returns TestingStreamMessage instance
             */
            public static create(properties?: lagrangian.testing.ITestingStreamMessage): lagrangian.testing.TestingStreamMessage;

            /**
             * Encodes the specified TestingStreamMessage message. Does not implicitly {@link lagrangian.testing.TestingStreamMessage.verify|verify} messages.
             * @param message TestingStreamMessage message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encode(message: lagrangian.testing.ITestingStreamMessage, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Encodes the specified TestingStreamMessage message, length delimited. Does not implicitly {@link lagrangian.testing.TestingStreamMessage.verify|verify} messages.
             * @param message TestingStreamMessage message or plain object to encode
             * @param [writer] Writer to encode to
             * @returns Writer
             */
            public static encodeDelimited(message: lagrangian.testing.ITestingStreamMessage, writer?: $protobuf.Writer): $protobuf.Writer;

            /**
             * Decodes a TestingStreamMessage message from the specified reader or buffer.
             * @param reader Reader or buffer to decode from
             * @param [length] Message length if known beforehand
             * @returns TestingStreamMessage
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decode(reader: ($protobuf.Reader|Uint8Array), length?: number): lagrangian.testing.TestingStreamMessage;

            /**
             * Decodes a TestingStreamMessage message from the specified reader or buffer, length delimited.
             * @param reader Reader or buffer to decode from
             * @returns TestingStreamMessage
             * @throws {Error} If the payload is not a reader or valid buffer
             * @throws {$protobuf.util.ProtocolError} If required fields are missing
             */
            public static decodeDelimited(reader: ($protobuf.Reader|Uint8Array)): lagrangian.testing.TestingStreamMessage;

            /**
             * Verifies a TestingStreamMessage message.
             * @param message Plain object to verify
             * @returns `null` if valid, otherwise the reason why it is not
             */
            public static verify(message: { [k: string]: any }): (string|null);

            /**
             * Creates a TestingStreamMessage message from a plain object. Also converts values to their respective internal types.
             * @param object Plain object
             * @returns TestingStreamMessage
             */
            public static fromObject(object: { [k: string]: any }): lagrangian.testing.TestingStreamMessage;

            /**
             * Creates a plain object from a TestingStreamMessage message. Also converts values to other types if specified.
             * @param message TestingStreamMessage
             * @param [options] Conversion options
             * @returns Plain object
             */
            public static toObject(message: lagrangian.testing.TestingStreamMessage, options?: $protobuf.IConversionOptions): { [k: string]: any };

            /**
             * Converts this TestingStreamMessage to JSON.
             * @returns JSON object
             */
            public toJSON(): { [k: string]: any };

            /**
             * Gets the default type url for TestingStreamMessage
             * @param [typeUrlPrefix] your custom typeUrlPrefix(default "type.googleapis.com")
             * @returns The default type url
             */
            public static getTypeUrl(typeUrlPrefix?: string): string;
        }
    }
}
