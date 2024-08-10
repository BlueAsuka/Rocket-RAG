import React, { useEffect, useState } from 'react';
import {
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  LoadingOutlined,
  DeleteOutlined,
  OpenAIOutlined,
  FormOutlined,
} from '@ant-design/icons';
import { Button, Layout, Menu, Avatar, Divider, theme, message } from 'antd';
import './css/App.css';

import Conversation from './Conversation';
import Conversation_Handler, { ConversationHeaders } from './Api_Handlers/Conversation_Handler';
import { ServerResponse } from './Api_Handlers/Fetch_Helper';


type MenuItem = {
  key: string;
  label: React.ReactNode;
  icon: React.ReactNode;
  onClick: () => void;
};

const { Header, Sider, Content } = Layout;

const App: React.FC = () => {
  const [collapsed, setCollapsed] = useState<boolean>(false);
  const [conversation, setConversation] = useState<string>("-1");
  const [menuItems, setMenuItems] = useState<MenuItem[]>([{ key: '1', icon: <LoadingOutlined/>, label: "Loading" , onClick: () => openMessage() }]);
  const [messageApi, contextHolder] = message.useMessage();
  const [messageOpenned, setMessageOpenned] = useState<boolean>(false);
  const key: string = 'updata';
  const openMessage = () => {
    console.log('openMessage', messageOpenned);
    messageApi.open({
      key,
      type: 'loading',
      content: 'Loading...',
    });
    setMessageOpenned(true);
  }

  const {
    token: { colorBgContainer, borderRadiusLG },
  } = theme.useToken();

  const deleteConversation = (conversationId: ConversationHeaders) => {
    Conversation_Handler.deleteConversation(conversationId).then((response : ServerResponse) => {
      console.log(response);
      if (response.status !== 200) {
        messageApi.error({ content: 'Error deleting conversation', key, duration: 2 });
        return;
      }
      messageApi.success({ content: 'Conversation deleted', key, duration: 2 });
      loadConversationsHeaders();
    }).catch((error : ServerResponse) => {
      console.log(error);
      messageApi.error({ content: 'Error deleting conversation', key, duration: 2 });
    });
  }

  const createConversation = () => {
    Conversation_Handler.createConversation().then((conversationHeaders : ConversationHeaders) => {
      setMenuItems([{ 
        key: conversationHeaders.id,
        label: <div> {conversationHeaders.title} <Button className='conversation_delete_btn' danger type="text" icon={<DeleteOutlined />} onClick={() => deleteConversation(conversationHeaders)} /></div>,
        onClick: () => setConversation(conversationHeaders.id),
        icon: <OpenAIOutlined /> },
        ...menuItems]);
      setConversation(conversationHeaders.id);
      messageApi.success({ content: 'Conversation created', key, duration: 2 });
    }).catch((error : ServerResponse) => {
      console.log(error);
      messageApi.error({ content: 'Error creating conversation', key, duration: 2 });
    });
  }

  const loadConversationsHeaders = () => {
    // Fetch and load the list of conversations
    // This is a placeholder for actual fetching logic
    Conversation_Handler.getConversationsHeaders().then((conversations : ConversationHeaders[]) => {
      setMenuItems(conversations.map((conversation) => {
        return {
          key: conversation.id,
          icon: <OpenAIOutlined />,
          label: <div> {conversation.title} <Button className='conversation_delete_btn' danger type="text" icon={<DeleteOutlined />} onClick={() => deleteConversation(conversation)} /></div>,
          onClick: () => setConversation(conversation.id),
        }
      }));
      messageApi.success({ content: 'Conversations loaded', key, duration: 2 });
    }).catch((error : ServerResponse) => {
      console.log(error);
      messageApi.error({ content: 'Error loading conversations', key, duration: 2 });
    });
  }

  useEffect(() => {
    loadConversationsHeaders();
  }, []);

  useEffect(() => {
    // Disable delete button if collapsed
    if (collapsed) {
      // Change the CSS of the delete buttons
      const elements = document.getElementsByClassName('conversation_delete_btn');
      for (let i = 0; i < elements.length; i++) {
        elements[i].setAttribute('style', 'display: none');
      }
    } else {
      // Change the CSS of the delete buttons
      const elements = document.getElementsByClassName('conversation_delete_btn');
      for (let i = 0; i < elements.length; i++) {
        elements[i].setAttribute('style', 'display: block');
      }
    }
  }
  , [collapsed, menuItems]);

  return (
    <Layout className='home_layout'>
      <Sider trigger={null} collapsible collapsed={collapsed} >
        <Avatar src={<img src={import.meta.env.VITE_IMG_LOGO} alt={import.meta.env.VITE_IMG_LOGO_ALT} />} className='avatar'/>
        <Divider />
        {contextHolder}
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[conversation]}
          items={[
            {
              key: '-1',
              label: 'Create conversation',
              icon: <FormOutlined />,
              onClick: () => createConversation(),
            },
            ...menuItems
          ]}
        />
      </Sider>
      <Layout className='home_layout'>
        <Header style={{ padding: 0, background: colorBgContainer }}>
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{
              fontSize: '16px',
              width: 64,
              height: 64,
            }}
          />
        </Header>
        <Content
          style={{
            margin: '24px 16px',
            padding: 24,
            minHeight: 280,
            background: colorBgContainer,
            borderRadius: borderRadiusLG,
          }}
        >
          {/* Terinary operator to check if a conversation is selected */}
          { conversation === "-1" ?  
            <h1> Select a conversation to start chatting </h1> :
            <Conversation conversationId={conversation.toString()} />
          }
        </Content>
      </Layout>
    </Layout>
  );
};

export default App;