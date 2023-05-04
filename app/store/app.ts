import { create } from "zustand";
import { persist } from "zustand/middleware";

import { type ChatCompletionResponseMessage } from "openai";
import {
  ControllerPool,
  requestChatStream,
  requestWithPrompt,
} from "../requests";
import { isMobileScreen, trimTopic } from "../utils";

import Locale from "../locales";
import { showToast } from "../components/ui-lib";

export type Message = ChatCompletionResponseMessage & {
  date: string;
  streaming?: boolean;
  isError?: boolean;
  id?: number;
};


export function createMessage(override: Partial<Message>): Message {
  return {
    id: Date.now(),
    date: new Date().toLocaleString(),
    role: "user",
    content: "",
    ...override,
  };
}

export enum SubmitKey {
  Enter = "Enter",
  CtrlEnter = "Ctrl + Enter",
  ShiftEnter = "Shift + Enter",
  AltEnter = "Alt + Enter",
  MetaEnter = "Meta + Enter",
}

export enum Theme {
  Auto = "auto",
  Dark = "dark",
  Light = "light",
}

export interface ChatConfig {
  historyMessageCount: number; // -1 means all
  compressMessageLengthThreshold: number;
  sendBotMessages: boolean; // send bot's message or not
  submitKey: SubmitKey;
  avatar: string;
  fontSize: number;
  theme: Theme;
  tightBorder: boolean;
  sendPreviewBubble: boolean;

  disablePromptHint: boolean;

  modelConfig: {
    model: string;
    temperature: number;
    max_tokens: number;
    presence_penalty: number;
  };
}

export type ModelConfig = ChatConfig["modelConfig"];

export const ROLES: Message["role"][] = ["system", "user", "assistant"];

const ENABLE_GPT4 = true;

export const ALL_MODELS = [
  {
    name: "gpt-4",
    available: ENABLE_GPT4,
  },
  {
    name: "gpt-4-0314",
    available: ENABLE_GPT4,
  },
  {
    name: "gpt-4-32k",
    available: ENABLE_GPT4,
  },
  {
    name: "gpt-4-32k-0314",
    available: ENABLE_GPT4,
  },
  {
    name: "gpt-3.5-turbo",
    available: true,
  },
  {
    name: "gpt-3.5-turbo-0301",
    available: true,
  },
];

export function limitNumber(
  x: number,
  min: number,
  max: number,
  defaultValue: number,
) {
  if (typeof x !== "number" || isNaN(x)) {
    return defaultValue;
  }

  return Math.min(max, Math.max(min, x));
}

export function limitModel(name: string) {
  return ALL_MODELS.some((m) => m.name === name && m.available)
    ? name
    : ALL_MODELS[4].name;
}

export const ModalConfigValidator = {
  model(x: string) {
    return limitModel(x);
  },
  max_tokens(x: number) {
    return limitNumber(x, 0, 32000, 2000);
  },
  presence_penalty(x: number) {
    return limitNumber(x, -2, 2, 0);
  },
  temperature(x: number) {
    return limitNumber(x, 0, 2, 1);
  },
};

const DEFAULT_CONFIG: ChatConfig = {
  historyMessageCount: 4,
  compressMessageLengthThreshold: 1000,
  sendBotMessages: true as boolean,
  submitKey: SubmitKey.CtrlEnter as SubmitKey,
  avatar: "1f603",
  fontSize: 14,
  theme: Theme.Auto as Theme,
  tightBorder: false,
  sendPreviewBubble: false,

  disablePromptHint: false,

  modelConfig: {
    model: "gpt-3.5-turbo",
    temperature: 0.7,
    max_tokens: 4000,
    presence_penalty: 0,
  },
};

export interface ChatStat {
  tokenCount: number;
  wordCount: number;
  charCount: number;
}

export interface ChatSession {
  id: number;
  topic: string;
  sendMemory: boolean;
  memoryPrompt: string;
  context: Message[];
  messages: Message[];
  stat: ChatStat;
  lastUpdate: string;
  lastSummarizeIndex: number;
}

const DEFAULT_TOPIC = Locale.Store.DefaultTopic;
export const BOT_HELLO: Message = createMessage({
  role: "assistant",
  content: Locale.Store.BotHello,
});

function createEmptySession(): ChatSession {
  const createDate = new Date().toLocaleString();

  return {
    id: Date.now(),
    topic: DEFAULT_TOPIC,
    sendMemory: true,
    memoryPrompt: "",
    context: [],
    messages: [],
    stat: {
      tokenCount: 0,
      wordCount: 0,
      charCount: 0,
    },
    lastUpdate: createDate,
    lastSummarizeIndex: 0,
  };
}

interface ChatStore {
  config: ChatConfig;
  sessions: ChatSession[];
  currentSessionIndex: number;
  clearSessions: () => void;
  removeSession: (index: number) => void;
  moveSession: (from: number, to: number) => void;
  selectSession: (index: number) => void;
  newSession: () => void;
  deleteSession: () => void;
  currentSession: () => ChatSession;
  onNewMessage: (message: Message) => void;
  onUserInput: (content: string) => Promise<void>;
  summarizeSession: () => void;
  updateStat: (message: Message) => void;
  updateCurrentSession: (updater: (session: ChatSession) => void) => void;
  updateMessage: (
    sessionIndex: number,
    messageIndex: number,
    updater: (message?: Message) => void,
  ) => void;
  resetSession: () => void;
  getMessagesWithMemory: () => Message[];
  getMemoryPrompt: () => Message;

  getConfig: () => ChatConfig;
  resetConfig: () => void;
  updateConfig: (updater: (config: ChatConfig) => void) => void;
  clearAllData: () => void;
}

function countMessages(msgs: Message[]) {
  return msgs.reduce((pre, cur) => pre + cur.content.length, 0);
}

const LOCAL_KEY = "chat-next-web-store";

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      sessions: [createEmptySession()],
      currentSessionIndex: 0,
      config: {
        ...DEFAULT_CONFIG,
      },

      clearSessions() {
        set(() => ({
          sessions: [createEmptySession()],
          currentSessionIndex: 0,
        }));
      },

      resetConfig() {
        set(() => ({ config: { ...DEFAULT_CONFIG } }));
      },

      getConfig() {
        return get().config;
      },

      updateConfig(updater) {
        const config = get().config;
        updater(config);
        set(() => ({ config }));
      },

      selectSession(index: number) {
        set({
          currentSessionIndex: index,
        });
      },

      removeSession(index: number) {
        set((state) => {
          let nextIndex = state.currentSessionIndex;
          const sessions = state.sessions;

          if (sessions.length === 1) {
            return {
              currentSessionIndex: 0,
              sessions: [createEmptySession()],
            };
          }

          sessions.splice(index, 1);

          if (nextIndex === index) {
            nextIndex -= 1;
          }

          return {
            currentSessionIndex: nextIndex,
            sessions,
          };
        });
      },

      moveSession(from: number, to: number) {
        set((state) => {
          const { sessions, currentSessionIndex: oldIndex } = state;

          // move the session
          const newSessions = [...sessions];
          const session = newSessions[from];
          newSessions.splice(from, 1);
          newSessions.splice(to, 0, session);

          // modify current session id
          let newIndex = oldIndex === from ? to : oldIndex;
          if (oldIndex > from && oldIndex <= to) {
            newIndex -= 1;
          } else if (oldIndex < from && oldIndex >= to) {
            newIndex += 1;
          }

          return {
            currentSessionIndex: newIndex,
            sessions: newSessions,
          };
        });
      },

      newSession() {
        set((state) => ({
          currentSessionIndex: 0,
          sessions: [createEmptySession()].concat(state.sessions),
        }));
      },

      deleteSession() {
        const deletedSession = get().currentSession();
        const index = get().currentSessionIndex;
        const isLastSession = get().sessions.length === 1;
        if (!isMobileScreen() || confirm(Locale.Home.DeleteChat)) {
          get().removeSession(index);
        }
        showToast(Locale.Home.DeleteToast, {
          text: Locale.Home.Revert,
          onClick() {
            set((state) => ({
              sessions: state.sessions
                .slice(0, index)
                .concat([deletedSession])
                .concat(state.sessions.slice(index + Number(isLastSession))),
            }));
          },
        });
      },

      currentSession() {
        let index = get().currentSessionIndex;
        const sessions = get().sessions;

        if (index < 0 || index >= sessions.length) {
          index = Math.min(sessions.length - 1, Math.max(0, index));
          set(() => ({ currentSessionIndex: index }));
        }

        const session = sessions[index];

        return session;
      },

      onNewMessage(message) {
        console.log("Get New Message: ", message);
        get().updateCurrentSession((session) => {
          session.lastUpdate = new Date().toLocaleString();
        });
        // get().updateStat(message);
        // get().summarizeSession();
      },

      async onUserInput(content) {
        const userMessage: Message = createMessage({
          role: "user",
          content,
        });

        const botMessage: Message = createMessage({
          role: "assistant",
          streaming: true,
        });

        // get recent messages
        const recentMessages = get().getMessagesWithMemory();
        const sendMessages = recentMessages.concat(userMessage);
        const sessionIndex = get().currentSessionIndex;
        const messageIndex = get().currentSession().messages.length + 1;

        // save user's and bot's message
        get().updateCurrentSession((session) => {
          session.messages.push(userMessage);
          session.messages.push(botMessage);
        });

        // make request
        console.log("[User Input] ", sendMessages);
        requestChatStream(sendMessages, {
          onMessage(content, done) {
            // stream response
            if (done) {
              botMessage.streaming = false;
              botMessage.content = content;
              get().onNewMessage(botMessage);
              ControllerPool.remove(
                sessionIndex,
                botMessage.id ?? messageIndex,
              );
            } else {
              botMessage.content = content;
              set(() => ({}));
            }
          },
          onError(error, statusCode) {
            if (statusCode === 401) {
              botMessage.content = Locale.Error.Unauthorized;
            } else {
              botMessage.content += "\n\n" + Locale.Store.Error;
            }
            botMessage.streaming = false;
            userMessage.isError = true;
            botMessage.isError = true;
            set(() => ({}));
            ControllerPool.remove(sessionIndex, botMessage.id ?? messageIndex);
          },
          onController(controller) {
            // collect controller for stop/retry
            ControllerPool.addController(
              sessionIndex,
              botMessage.id ?? messageIndex,
              controller,
            );
          },
          filterBot: !get().config.sendBotMessages,
          modelConfig: get().config.modelConfig,
        });
      },

      getMemoryPrompt() {
        const session = get().currentSession();

        return {
          role: "system",
          content: Locale.Store.Prompt.History(session.memoryPrompt),
          date: "",
        } as Message;
      },
      
      getMessagesWithMemory() {
        const session = get().currentSession();
        const config = get().config;
        const messages = session.messages.filter((msg) => !msg.isError);
        const n = messages.length;
        // messages for few shots
        const exampleMessages: Message[] = [
          {"role":"user","content":"I need a resource to find ONSSA-approved pesticides for my crops in Morocco. Can you help?"},{"role":"assistant","content":"Certainly! The ONSSA index of pesticides is a valuable resource that lists approved agricultural pesticide products in Morocco. You can access the index here: http://eservice.onssa.gov.ma/IndPesticide.aspx\\n\\nThe index allows you to search for pesticide information by commercial name, active ingredient, category, use, crop, pest, holder, supplier, and multi-criteria search. The data is presented in alphabetical order within each section."},{"role":"user","content":"I want to know if a pesticide is allowed for agricultural use, can you help?"},{"role":"assistant","content":"The ONSSA (National Office of Sanitary Safety of Food Products) maintains an index of approved pesticides for agricultural use in Morocco. You can find the index at the following link: http://eservice.onssa.gov.ma/IndPesticide.aspx\\n\\nThis comprehensive index provides information on approved pesticide products, allowing you to search by various criteria such as commercial name, active ingredient, category, use, crop, pest, holder, and supplier. It's a helpful resource for finding the right pesticide for your needs."},{"role":"user","content":"What is DeepLeaf, and what services do they offer?"},{"role":"assistant","content":"DeepLeaf is an agricultural deep learning lab that focuses on using AI technology to revolutionize crop growth. Their services include:\\n\\n1. Real-time disease detection: Using your smartphone camera, you can detect diseases in your crops without the need for expensive hardware or sensors.\\n2. Insightful reports and preventive analysis: DeepLeaf's AI provides alerts and expert-recommended solutions to protect your crops.\\n3. Cost reduction: By minimizing the need for expert consultations, you can increase your profits.\\n\\nTo learn more about DeepLeaf, visit their website at https://deepleaf.io or contact them at contact@deepleaf.io or +212708765350."},{"role":"user","content":"How can DeepLeaf help me with my agricultural needs?"},{"role":"assistant","content":"DeepLeaf offers a range of innovative solutions for your agricultural needs, all accessible via WhatsApp for easy access and a seamless user experience. The solutions include:\\n\\n1. Greenius: A GPT-3.5-turbo AI assistant that specializes in agriculture, providing advice, consultation, and expertise on various topics, including plant disease diagnosis and treatment suggestions.\\n\\n2. FitoFoto: A WhatsApp chatbot that detects plant diseases from images using deep learning and suggests treatments.\\n\\n3. Agrigo: An autonomous rover equipped with a multispectral camera that automates plant disease detection and pesticide spraying, reducing chemical usage by up to 60%.\\n For more information, visit their website at https://deepleaf.io, email them at contact@deepleaf.io, or call +212708765350."},{"role":"user","content":"I have weird yellow leaves in my tomato crops lately, when I grab a leaf it's crunchy and the tomato fruits are blocked it doesn't grow anymore."},{"role":"assistant","content":"Yellow leaves in tomato crops can be caused by a variety of factors, including nutrient deficiencies, pests, or diseases. The crunchy leaves and stunted fruit growth suggest that your plants may be affected by a disease. Here are some possible causes and treatments:\n\nTomato mosaic virus: This viral disease causes yellowing and mottling of leaves, and stunted growth of fruits. Unfortunately, there is no cure for this disease, and infected plants should be removed and destroyed to prevent its spread.\nFusarium wilt: This fungal disease causes yellowing and wilting of leaves, and stunted growth of fruits. Infected plants should be removed and destroyed, and crop rotation should be practiced to prevent the disease from spreading.\nBacterial spot: This bacterial disease causes yellowing of leaves, as well as black spots on leaves and fruits. Copper-based fungicides can be used to control the disease, and infected plant debris should be removed and destroyed.\nNutrient deficiency: A lack of nutrients like nitrogen, magnesium, or iron can cause yellowing of leaves. Check the soil pH and nutrient levels, and adjust fertilization accordingly.\nTo accurately diagnose the cause of the problem and receive tailored treatment recommendations, you can use FitoFoto, a WhatsApp chatbot developed by DeepLeaf. FitoFoto uses deep learning to detect plant diseases from images and suggests appropriate treatments. Simply send a clear image of the affected plant to FitoFoto's WhatsApp number (+212 678-995138).\n\nIf you need further assistance or support, feel free to contact DeepLeaf at contact@deepleaf.io or call +212708765350."},{"role":"user","content":"What should I do if I suspect my plants have a disease, but I'm not sure?"},{"role":"assistant","content":"If you suspect your plant has a disease, it's important to accurately diagnose the cause of the problem to apply the correct treatment. Can you please provide some more details? For example, what is the color of the leaves, how does it feel in your hand, what is the weather like, and are the soil moisture and NPK values good?\\n\\nOnce you have this information, you can use FitoFoto by DeepLeaf to identify the plant disease and get appropriate treatments. Simply send a clear image of the affected plant to FitoFoto's WhatsApp number (+212 678-995138).\\n\\nIf you need further assistance or support, feel free to contact DeepLeaf at contact@deepleaf.io or call +212708765350. We're here to help you keep your crops healthy and thriving!"},{"role":"user","content":"My plant's leaves are turning yellow. What should I do?"},{"role":"assistant","content":"Thank you for reaching out! To accurately diagnose the issue, please provide additional details such as the color of the leaves and the texture of the plant. Additionally, please let me know about the soil moisture and NPK values. Once you have this information, you can use FitoFoto by DeepLeaf to identify the plant disease and get appropriate treatments. Simply send a clear image of the affected plant to FitoFoto's WhatsApp number (+212 678-995138).\\n\\nIf you need further assistance or support, do not hesitate to contact DeepLeaf at contact@deepleaf.io or call +212708765350. We are here to help you keep your crops healthy and thriving!"},{"role":"user","content":"My tomato plant leaves are turning brown. What should I do?"},{"role":"assistant","content":"Thank you for reaching out! To accurately diagnose the issue, please provide additional details such as the weather, soil moisture, and NPK values. Once you have this information, you can use FitoFoto by DeepLeaf to identify the plant disease and get appropriate treatments. Simply send a clear image of the affected plant to FitoFoto's WhatsApp number (+212 678-995138).\\n\\nIf you need further assistance or support, do not hesitate to contact DeepLeaf at contact@deepleaf.io or call +212708765350. We are here to help you keep your crops healthy and thriving!"},{"role":"user","content":"My cucumber plant is not growing well. What should I do?"},{"role":"assistant","content":"Thank you for reaching out! To accurately diagnose the issue, please provide additional details such as the color of the leaves and the texture of the plant. Additionally, please let me know about the soil moisture and NPK values. Once you have this information, you can use FitoFoto by DeepLeaf to identify the plant disease and get appropriate treatments. Simply send a clear image of the affected plant to FitoFoto's WhatsApp number (+212 678-995138).\\n\\nIf you need further assistance or support, do not hesitate to contact DeepLeaf at contact@deepleaf.io or call +212708765350. We are here to help you keep your crops healthy and thriving!"},{"role":"user","content":"How can I prevent pests from damaging my crops?"},{"role":"assistant","content":"Preventing pests from damaging your crops is an important aspect of agriculture. Some methods for pest control include natural predators, crop rotation, and the use of insecticides.\\n\\nTo prevent pests, make sure to keep your fields and equipment clean, and inspect your crops regularly for signs of infestation. It's also important to research the specific pests in your area and their life cycles so you can take preventative measures before they become a problem. If you suspect that your crops have been affected by pests, you can use FitoFoto by DeepLeaf to identify the problem and get appropriate treatments. Simply send a clear image of the affected plant to FitoFoto's WhatsApp number (+212 678-995138).\\n\\nIf you need further assistance or support, do not hesitate to contact DeepLeaf at contact@deepleaf.io or call +212708765350. We are here to help you keep your crops healthy and thriving!"},{"role":"user","content":"How can I effectively manage irrigation water for my crops?"},{"role":"assistant","content":"Effective irrigation water management is crucial for maintaining healthy and productive crops. Some methods for irrigation water management include using drip or micro-irrigation systems, monitoring soil moisture levels, and adjusting irrigation frequency and duration according to the specific needs of your crops. DeepLeaf offers H2OM, an AI-powered platform that uses machine learning algorithms and sensors to optimize irrigation water usage and reduce water waste. It's also important to use high-quality irrigation water and properly manage any potential sources of contamination to prevent plant diseases.\\n\\nIf you need further assistance or support, do not hesitate to contact DeepLeaf at contact@deepleaf.io or call +212708765350. We are here to help you manage your irrigation water effectively and promote healthy crop growth!"},{"role":"user","content":"I'm a farmer and I'm struggling to keep up with the rising prices of pesticides. What can I do to reduce my pesticide costs?"},{"role":"assistant","content":"The rising prices of pesticides can be a significant financial burden for farmers. DeepLeaf offers Agrigo, an autonomous rover with a multispectral camera that can automate plant disease detection and pesticide spraying to reduce up to 60% of chemical use. By reducing pesticide use, you can lower input costs and ultimately reduce your expenses.\\n\\nIn addition to using Agrigo, you can also consider implementing integrated pest management (IPM) practices, which involve using a combination of methods to control pests and diseases, including natural predators, crop rotation, and cultural practices. IPM can help reduce the need for pesticides and ultimately lower your costs. DeepLeaf also offers FitoFoto, a WhatsApp chatbot that can detect plant diseases from images using deep learning and suggest treatments. It can also provide insights on the health and growth of your crops.\\n\\nBy using these solutions and strategies, you can reduce your pesticide costs and increase your profitability as a farmer. If you need further guidance or support, do not hesitate to contact DeepLeaf at contact@deepleaf.io or call +212708765350."},{"role":"user","content":"I have Tuta Absoluta"},{"role":"assistant","content":"I'm sorry to hear that you have Tuta Absoluta, which is a destructive pest that can cause significant damage to tomato crops. Here are some steps you can take to manage Tuta Absoluta infestations: \nMonitor your crops: Regularly inspect your tomato plants using FitoFoto by DeepLeaf to identify the plant disease and get appropriate treatments. Simply send a clear image of the affected plant to FitoFoto's WhatsApp number (+212 678-995138) \\n\nUse biological control: There are various biological control methods that can help manage Tuta Absoluta, including using predatory insects such as Nesidiocoris tenuis, Trichogramma wasps, and Bacillus thuringiensis-based biopesticides. \\n\nUse chemical control: If necessary, chemical control methods can also be used to manage Tuta Absoluta infestations. However, it is important to use pesticides responsibly and carefully follow label instructions to avoid harming beneficial insects and pollinators. \\n\nCrop rotation: Practicing crop rotation can help prevent Tuta Absoluta infestations by disrupting the pest's life cycle and reducing the availability of host plants. \n\\n\nIf you need further assistance or support, you can use FitoFoto by DeepLeaf to identify the plant disease and get appropriate treatments. Simply send a clear image of the affected plant to FitoFoto's WhatsApp number (+212 678-995138). If you have any other questions or concerns, please do not hesitate to contact DeepLeaf at contact@deepleaf.io or call +212708765350.\n"},{"role":"user","content":"hello what are you"},{"role":"assistant","content":"Hello! I am Greenius, an AI assistant developed by DeepLeaf. My expertise lies in the field of agriculture, where I offer advice, consultation, and support on a wide range of topics, including the diagnosis and treatment of plant diseases."},
        ];
        const currentDate = new Date().toISOString();
        const messagesWithDate = exampleMessages.map((msg) => ({
          ...msg,
          date: currentDate,
        }));

        const context = session.context.slice();

        if (
          session.sendMemory &&
          session.memoryPrompt &&
          session.memoryPrompt.length > 0
        ) {
          const memoryPrompt = get().getMemoryPrompt();
          context.push(memoryPrompt);
        }

        const recentMessages = context.concat(exampleMessages, messages.slice(Math.max(0, n - config.historyMessageCount)));


        //Add system prompt
        if (process.env.NEXT_PUBLIC_SYSTEM_PROMPT != undefined) {
            const systemMessage: Message = {
              content: `${process.env.NEXT_PUBLIC_SYSTEM_PROMPT}`,
              role: "system",
              date: "",
            };
          
          recentMessages.unshift(systemMessage);
          
        }
        
        // const recentMessages = context.concat(
        //   messages.slice(Math.max(0, n - config.historyMessageCount)),
        // );

        return recentMessages;
      },

      updateMessage(
        sessionIndex: number,
        messageIndex: number,
        updater: (message?: Message) => void,
      ) {
        const sessions = get().sessions;
        const session = sessions.at(sessionIndex);
        const messages = session?.messages;
        updater(messages?.at(messageIndex));
        set(() => ({ sessions }));
      },

      resetSession() {
        get().updateCurrentSession((session) => {
          session.messages = [];
          session.memoryPrompt = "";
        });
      },

      summarizeSession() {
        const session = get().currentSession();

        // should summarize topic after chating more than 50 words
        const SUMMARIZE_MIN_LEN = 50;
        if (
          session.topic === DEFAULT_TOPIC &&
          countMessages(session.messages) >= SUMMARIZE_MIN_LEN
        ) {
          requestWithPrompt(session.messages, Locale.Store.Prompt.Topic).then(
            (res) => {
              get().updateCurrentSession(
                (session) =>
                  (session.topic = res ? trimTopic(res) : DEFAULT_TOPIC),
              );
            },
          );
        }

        const config = get().config;
        let toBeSummarizedMsgs = session.messages.slice(
          session.lastSummarizeIndex,
        );

        const historyMsgLength = countMessages(toBeSummarizedMsgs);

        if (historyMsgLength > get().config?.modelConfig?.max_tokens ?? 4000) {
          const n = toBeSummarizedMsgs.length;
          toBeSummarizedMsgs = toBeSummarizedMsgs.slice(
            Math.max(0, n - config.historyMessageCount),
          );
        }

        // add memory prompt
        toBeSummarizedMsgs.unshift(get().getMemoryPrompt());

        const lastSummarizeIndex = session.messages.length;

        console.log(
          "[Chat History] ",
          toBeSummarizedMsgs,
          historyMsgLength,
          config.compressMessageLengthThreshold,
        );

        if (historyMsgLength > config.compressMessageLengthThreshold) {
          requestChatStream(
            toBeSummarizedMsgs.concat({
              role: "system",
              content: Locale.Store.Prompt.Summarize,
              date: "",
            }),
            {
              filterBot: false,
              onMessage(message, done) {
                session.memoryPrompt = message;
                if (done) {
                  console.log("[Memory] ", session.memoryPrompt);
                  session.lastSummarizeIndex = lastSummarizeIndex;
                }
              },
              onError(error) {
                console.error("[Summarize] ", error);
              },
            },
          );
        }
      },

      updateStat(message) {
        get().updateCurrentSession((session) => {
          session.stat.charCount += message.content.length;
          // TODO: should update chat count and word count
        });
      },

      updateCurrentSession(updater) {
        const sessions = get().sessions;
        const index = get().currentSessionIndex;
        updater(sessions[index]);
        set(() => ({ sessions }));
      },

      clearAllData() {
        if (confirm(Locale.Store.ConfirmClearAll)) {
          localStorage.clear();
          location.reload();
        }
      },
    }),
    {
      name: LOCAL_KEY,
      version: 1.2,
      migrate(persistedState, version) {
        const state = persistedState as ChatStore;

        if (version === 1) {
          state.sessions.forEach((s) => (s.context = []));
        }

        if (version < 1.2) {
          state.sessions.forEach((s) => (s.sendMemory = true));
        }

        return state;
      },
    },
  ),
);
