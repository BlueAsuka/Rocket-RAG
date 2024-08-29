import Fetch_Helper, { ServerResponse } from "./Fetch_Helper";

export type Message = {
    id: string;
    text: string;
    sender: string;
}

export type Attachments = {
    id: string;
    name: string;
    url: string;
}

export type ConversationHeaders = {
    id: string;
    title: string;
    createdAt: string;
}

export type ConversationData = {
    header: ConversationHeaders;
    messages: Message[];
    attachments: Attachments[];
}

const Conversation_Handler = {
    createConversation: async (): Promise<ConversationHeaders> => {
        // Cast to ConversationHeaders
        return await Fetch_Helper.ApiPOSTCallNoTimeout('/conversation', JSON.parse("{}")) as ConversationHeaders;
    },
    getConversationsHeaders: async (): Promise<ConversationHeaders[]> => {
        return await Fetch_Helper.ApiGETCallNoData('/conversationHeaders') as ConversationHeaders[];
    },
    deleteConversation: async (header :ConversationHeaders): Promise<ServerResponse> => {
        return await Fetch_Helper.ApiDELETECall('/conversation/' + header.id, JSON.parse("{}")) as ServerResponse;
    },
    loadFullConversation: async (header :ConversationHeaders): Promise<ConversationData> => {
        return await Fetch_Helper.ApiGETCallNoData('/conversationFull/' + header.id) as ConversationData;
    },
    sendMessage: async (header :ConversationHeaders, message: Message): Promise<ConversationData> => {
        return await Fetch_Helper.ApiPOSTCall('/conversationMessage', JSON.parse(JSON.stringify({ header, message }))) as ConversationData;
    },
    sendAttachment: async (header :ConversationHeaders, attachment: Attachments): Promise<ServerResponse> => {
        return await Fetch_Helper.ApiPOSTCall('/conversationAttachment', JSON.parse(JSON.stringify({ header, attachment }))) as ServerResponse;
    }
}

export default Conversation_Handler;