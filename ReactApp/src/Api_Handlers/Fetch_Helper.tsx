import { notification } from 'antd';

const addr: string = import.meta.env.VITE_BACKEND_URL || '';

// Import the types from Conversation_Handler.tsx
import { ConversationHeaders, Attachments, ConversationData, Message } from './Conversation_Handler';

interface FetchOptions {
  method: string;
  dataToUse?: JSON | null;
  timeout?: boolean;
}

export type ServerResponse = {
  status: number;
  message: string;
}

// Create a type that represents the server response (based on Conversation_Handler.tsx)
export type FetchResponse = Message | ConversationHeaders | ConversationHeaders[] | Attachments | ConversationData | ServerResponse;

const fetchData = async (apiRoute: string, options: FetchOptions): Promise<FetchResponse> => {
  const { method, dataToUse = null, timeout = true } = options;
  let timeoutId: NodeJS.Timeout | null = null;

  if (timeout) {
    timeoutId = setTimeout(() => {
      notification.error({
        message: 'Error',
        description: 'Error during the request',
      });
    }, 10000);
  }

  const requestOptions: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json; charset=UTF-8',
    },
    body: dataToUse ? JSON.stringify(dataToUse) : undefined,
  };

  try {
    const response = await fetch(`${addr}${apiRoute}`, requestOptions);
    if (!response.ok) {
      return { status: response.status, message: response.statusText };
    }
    return await response.json();
  } catch (error) {
    return { status: 500, message: (error as Error).message };
  } finally {
    if (timeout && timeoutId) {
      clearTimeout(timeoutId);
    }
  }
};

export const ApiGETCall = async (apiRoute: string, data: string): Promise<FetchResponse> => {
  return await fetchData(apiRoute + data, { method: 'GET' });
};

export const ApiGETCallNoData = async (apiRoute: string): Promise<FetchResponse> => {
  return await fetchData(apiRoute, { method: 'GET' });
};

export const ApiPOSTCall = async (apiRoute: string, dataToUse: JSON): Promise<FetchResponse> => {
  return await fetchData(apiRoute, { method: 'POST', dataToUse });
};

export const ApiPOSTCallNoTimeout = async (apiRoute: string, dataToUse: JSON): Promise<FetchResponse> => {
  return await fetchData(apiRoute, { method: 'POST', dataToUse, timeout: false });
};

export const ApiPUTCall = async (apiRoute: string, dataToUse: JSON): Promise<FetchResponse> => {
  return await fetchData(apiRoute, { method: 'PUT', dataToUse });
};

export const ApiDELETECall = async (apiRoute: string, dataToUse: JSON): Promise<FetchResponse> => {
  return await fetchData(apiRoute, { method: 'DELETE', dataToUse });
};

export const ApiDELETECallNoData = async (apiRoute: string): Promise<FetchResponse> => {
  return await fetchData(apiRoute, { method: 'DELETE' });
};


// Exporting all functions as a single object for easy import
export default {
  ApiGETCall,
  ApiGETCallNoData,
  ApiPOSTCall,
  ApiPOSTCallNoTimeout,
  ApiPUTCall,
  ApiDELETECall,
  ApiDELETECallNoData,
};
