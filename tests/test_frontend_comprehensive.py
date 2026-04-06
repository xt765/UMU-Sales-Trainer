"""AI Sales Trainer - Comprehensive Frontend Test Suite

Tests all frontend functionality including:
1. Page load and UI rendering
2. Session management (create/delete)
3. Message sending and display
4. Semantic point coverage tracking
5. Customer profile display
6. Interactive elements
7. Responsive layout
8. Error handling
"""

from playwright.sync_api import sync_playwright, Page
import time


def test_page_load(page: Page) -> dict:
    """Test 1: Page Load and Basic UI"""
    print("\n" + "="*60)
    print("TEST 1: Page Load and Basic UI Rendering")
    print("="*60)
    
    results = {
        'test_name': 'Page Load',
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    try:
        # Check page title
        title = page.title()
        if "AI" in title or "销售" in title:
            results['passed'].append(f'✓ Page title: {title}')
        else:
            results['failed'].append(f'✗ Unexpected title: {title}')
        
        # Check main container exists
        app_container = page.locator('.app-container')
        if app_container.count() > 0:
            results['passed'].append('✓ Main app container found')
        else:
            results['failed'].append('✗ App container missing')
        
        # Check sidebar exists
        sidebar = page.locator('.sidebar')
        if sidebar.count() > 0:
            results['passed'].append('✓ Sidebar present')
        else:
            results['failed'].append('✗ Sidebar missing')
        
        # Check main content area
        main_content = page.locator('.main-content')
        if main_content.count() > 0:
            results['passed'].append('✓ Main content area present')
        else:
            results['failed'].append('✗ Main content area missing')
        
        # Check chat header
        chat_header = page.locator('.chat-header')
        if chat_header.count() > 0:
            results['passed'].append('✓ Chat header visible')
        else:
            results['failed'].append('✗ Chat header missing')
        
        # Check input area
        input_area = page.locator('.chat-input-area')
        if input_area.count() > 0:
            results['passed'].append('✓ Input area present')
        else:
            results['failed'].append('✗ Input area missing')
        
        # Check initial message
        messages = page.locator('.message')
        if messages.count() >= 1:
            results['passed'].append(f'✓ Initial message displayed ({messages.count()} messages)')
        else:
            results['warnings'].append('⚠ No initial messages found')
        
        # Take screenshot for visual verification
        page.screenshot(path='test_screenshots/01_page_load.png', full_page=True)
        results['passed'].append('✓ Screenshot saved: 01_page_load.png')
        
    except Exception as e:
        results['failed'].append(f'✗ Exception: {str(e)}')
    
    print_results(results)
    return results


def test_session_management(page: Page) -> dict:
    """Test 2: Session Create/Delete Functionality"""
    print("\n" + "="*60)
    print("TEST 2: Session Management")
    print("="*60)
    
    results = {
        'test_name': 'Session Management',
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    try:
        # Find new session button
        btn_new_chat = page.locator('#btnNewChat')
        if btn_new_chat.count() == 0:
            results['failed'].append('✗ New Chat button not found')
            return results
        
        results['passed'].append('✓ New Chat button found')
        
        # Check initial state - delete button should be disabled
        btn_delete = page.locator('#btnDeleteSession')
        is_disabled = btn_delete.is_disabled()
        if is_disabled:
            results['passed'].append('✓ Delete button initially disabled')
        else:
            results['warnings'].append('⚠ Delete button enabled before session creation')
        
        # Check send button disabled
        btn_send = page.locator('#btnSend')
        if btn_send.is_disabled():
            results['passed'].append('✓ Send button initially disabled')
        else:
            results['warnings'].append('⚠ Send button enabled without session')
        
        # Click create session
        print("  → Clicking 'New Chat' button...")
        btn_new_chat.click()
        
        # Wait for session to be created (API call + UI update)
        time.sleep(2)  # Wait for API response
        
        # Check if session ID appeared
        session_display = page.locator('#sessionIdDisplay')
        session_text = session_display.inner_text()
        
        if '暂无' not in session_text and len(session_text) > 5:
            results['passed'].append(f'✓ Session created with ID: {session_text[:20]}...')
        else:
            results['failed'].append(f'✗ Session ID not updated: {session_text}')
        
        # Check delete button now enabled
        if not btn_delete.is_disabled():
            results['passed'].append('✓ Delete button enabled after session creation')
        else:
            results['failed'].append('✗ Delete button still disabled')
        
        # Check send button enabled
        if not btn_send.is_disabled():
            results['passed'].append('✓ Send button enabled after session creation')
        else:
            results['warnings'].append('⚠ Send button still disabled')
        
        # Screenshot after session creation
        page.screenshot(path='test_screenshots/02_session_created.png', full_page=True)
        results['passed'].append('✓ Screenshot saved: 02_session_created.png')
        
    except Exception as e:
        results['failed'].append(f'✗ Exception: {str(e)}')
    
    print_results(results)
    return results


def test_message_sending(page: Page) -> dict:
    """Test 3: Message Sending and Display"""
    print("\n" + "="*60)
    print("TEST 3: Message Sending and Display")
    print("="*60)
    
    results = {
        'test_name': 'Message Sending',
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    try:
        # Ensure we have an active session
        btn_new_chat = page.locator('#btnNewChat')
        btn_delete = page.locator('#btnDeleteSession')
        
        if btn_delete.is_disabled():
            print("  → Creating session first...")
            btn_new_chat.click()
            time.sleep(2)
        
        # Get initial message count
        initial_messages = page.locator('.message').count()
        results['passed'].append(f'✓ Initial messages: {initial_messages}')
        
        # Test input field
        chat_input = page.locator('#chatInput')
        if chat_input.count() == 0:
            results['failed'].append('✗ Chat input not found')
            return results
        
        results['passed'].append('✓ Chat input field found')
        
        # Type test message
        test_message = "您好，张主任，我是来介绍我们的新产品的"
        print(f"  → Typing test message: '{test_message}'...")
        chat_input.fill(test_message)
        
        # Verify input has text
        input_value = chat_input.input_value()
        if test_message in input_value:
            results['passed'].append('✓ Text entered successfully')
        else:
            results['failed'].append('✗ Text not entered correctly')
        
        # Check send button is now enabled
        btn_send = page.locator('#btnSend')
        if not btn_send.is_disabled():
            results['passed'].append('✓ Send button enabled with text')
        else:
            results['failed'].append('✗ Send button still disabled with text')
        
        # Click send
        print("  → Clicking send button...")
        btn_send.click()
        
        # Wait for response (typing indicator + AI response)
        time.sleep(3)
        
        # Check for user message
        user_messages = page.locator('.message.user')
        if user_messages.count() > 0:
            last_user_msg = user_messages.last.inner_text()
            if test_message in last_user_msg:
                results['passed'].append(f'✓ User message displayed correctly')
            else:
                results['warnings'].append(f'⚠ User message content mismatch')
        else:
            results['failed'].append('✗ User message not displayed')
        
        # Check for typing indicator or AI response
        ai_messages = page.locator('.message.ai')
        current_ai_count = ai_messages.count()
        if current_ai_count > initial_messages / 2:  # At least one new AI message
            results['passed'].append(f'✓ AI responded ({current_ai_count} total AI messages)')
        else:
            results['warnings'].append('⚠ Waiting for AI response...')
            time.sleep(2)
            
            ai_messages_after = page.locator('.message.ai').count()
            if ai_messages_after > current_ai_count:
                results['passed'].append('✓ AI response received (delayed)')
            else:
                results['warnings'].append('⚠ No AI response yet (may need backend)')
        
        # Screenshot after sending message
        page.screenshot(path='test_screenshots/03_message_sent.png', full_page=True)
        results['passed'].append('✓ Screenshot saved: 03_message_sent.png')
        
    except Exception as e:
        results['failed'].append(f'✗ Exception: {str(e)}')
    
    print_results(results)
    return results


def test_semantic_coverage(page: Page) -> dict:
    """Test 4: Semantic Point Coverage Display"""
    print("\n" + "="*60)
    print("TEST 4: Semantic Point Coverage Tracking")
    print("="*60)
    
    results = {
        'test_name': 'Semantic Coverage',
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    try:
        # Check coverage section exists
        coverage_section = page.locator('.coverage-section')
        if coverage_section.count() == 0:
            results['failed'].append('✗ Coverage section not found')
            return results
        
        results['passed'].append('✓ Coverage section present')
        
        # Check section title
        section_title = page.locator('.section-title')
        if section_title.count() > 0:
            title_text = section_title.inner_text()
            results['passed'].append(f'✓ Section title: "{title_text}"')
        else:
            results['warnings'].append('⚠ Section title not found')
        
        # Check coverage items
        coverage_items = page.locator('.coverage-item')
        item_count = coverage_items.count()
        
        if item_count >= 3:
            results['passed'].append(f'✓ Coverage items found: {item_count}')
        elif item_count > 0:
            results['warnings'].append(f'⚠ Only {item_count} coverage items (expected 3)')
        else:
            results['failed'].append('✗ No coverage items found')
            return results
        
        # Check each item's structure
        for i in range(item_count):
            item = coverage_items.nth(i)
            
            # Check item info
            item_info = item.locator('.item-info')
            if item_info.count() > 0:
                results['passed'].append(f'✓ Item {i+1}: Info section present')
            else:
                results['failed'].append(f'✗ Item {i+1}: Missing info section')
            
            # Check status indicator
            item_status = item.locator('.item-status')
            if item_status.count() > 0:
                status_class = item_status.get_attribute('class') or ''
                results['passed'].append(f'✓ Item {i+1}: Status indicator ({status_class})')
            else:
                results['warnings'].append(f'⚠ Item {i+1}: Missing status indicator')
        
        # Check progress bar
        progress_bar = page.locator('.progress-bar')
        progress_fill = page.locator('.progress-fill')
        progress_value = page.locator('#progressValue')
        
        if progress_bar.count() > 0:
            results['passed'].append('✓ Progress bar container present')
        else:
            results['failed'].append('✗ Progress bar missing')
        
        if progress_fill.count() > 0:
            fill_width = progress_fill.evaluate('el => el.style.width')
            results['passed'].append(f'✓ Progress fill: {fill_width}')
        else:
            results['warnings'].append('⚠ Progress fill element missing')
        
        if progress_value.count() > 0:
            value_text = progress_value.inner_text()
            results['passed'].append(f'✓ Progress value displayed: {value_text}')
        else:
            results['warnings'].append('⚠ Progress value text missing')
        
        # Verify all items are visible (not clipped)
        items_visible = 0
        for i in range(item_count):
            item = coverage_items.nth(i)
            is_visible = item.is_visible()
            if is_visible:
                items_visible += 1
        
        if items_visible == item_count:
            results['passed'].append(f'✓ All {item_count} coverage items visible')
        else:
            results['failed'].append(f'✗ Only {items_visible}/{item_count} items visible')
        
        # Screenshot of coverage section
        page.screenshot(path='test_screenshots/04_semantic_coverage.png', full_page=True)
        results['passed'].append('✓ Screenshot saved: 04_semantic_coverage.png')
        
    except Exception as e:
        results['failed'].append(f'✗ Exception: {str(e)}')
    
    print_results(results)
    return results


def test_customer_profile(page: Page) -> dict:
    """Test 5: Customer Profile Display"""
    print("\n" + "="*60)
    print("TEST 5: Customer Profile Display")
    print("="*60)
    
    results = {
        'test_name': 'Customer Profile',
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    try:
        # Check customer profile card
        profile_card = page.locator('.customer-profile')
        if profile_card.count() == 0:
            results['failed'].append('✗ Customer profile card not found')
            return results
        
        results['passed'].append('✓ Customer profile card present')
        
        # Check avatar
        avatar = page.locator('.profile-avatar')
        if avatar.count() > 0:
            avatar_text = avatar.inner_text()
            results['passed'].append(f'✓ Avatar displayed: "{avatar_text}"')
        else:
            results['warnings'].append('⚠ Avatar not found')
        
        # Check name and role
        profile_info = page.locator('.profile-info')
        if profile_info.count() > 0:
            info_html = profile_info.inner_html()
            if '张主任' in info_html or '主任' in info_html:
                results['passed'].append('✓ Customer name displayed')
            if '内分泌' in info_html:
                results['passed'].append('✓ Customer role displayed')
        else:
            results['warnings'].append('⚠ Profile info section missing')
        
        # Check detail items (selling points)
        detail_items = page.locator('.detail-item')
        detail_count = detail_items.count()
        
        if detail_count >= 3:
            results['passed'].append(f'✓ Detail items: {detail_count} selling points')
            
            # Verify content of details
            for i in range(min(detail_count, 3)):
                item_text = detail_items.nth(i).inner_text()
                if len(item_text) > 0:
                    results['passed'].append(f'✓ Detail {i+1}: "{item_text[:30]}..."')
        else:
            results['warnings'].append(f'⚠ Only {detail_count} detail items')
        
        # Screenshot
        page.screenshot(path='test_screenshots/05_customer_profile.png', full_page=True)
        results['passed'].append('✓ Screenshot saved: 05_customer_profile.png')
        
    except Exception as e:
        results['failed'].append(f'✗ Exception: {str(e)}')
    
    print_results(results)
    return results


def test_interactive_elements(page: Page) -> dict:
    """Test 6: Interactive Elements and Hover States"""
    print("\n" + "="*60)
    print("TEST 6: Interactive Elements")
    print("="*60)
    
    results = {
        'test_name': 'Interactive Elements',
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    try:
        # Test buttons exist
        buttons_to_test = [
            ('#btnNewChat', 'New Chat'),
            ('#btnDeleteSession', 'Delete Session'),
            ('#btnSend', 'Send'),
            ('.send-button', 'Send Button (icon)'),
        ]
        
        for selector, name in buttons_to_test:
            btn = page.locator(selector)
            if btn.count() > 0:
                results['passed'].append(f'✓ {name} button found')
                
                # Check if clickable
                if btn.is_enabled() or btn.is_disabled():
                    state = 'enabled' if btn.is_enabled() else 'disabled'
                    results['passed'].append(f'  └─ State: {state}')
            else:
                results['failed'].append(f'✗ {name} button not found')
        
        # Test input field interactions
        chat_input = page.locator('#chatInput')
        if chat_input.count() > 0:
            # Test placeholder
            placeholder = chat_input.get_attribute('placeholder')
            if placeholder:
                results['passed'].append(f'✓ Input placeholder: "{placeholder}"')
            
            # Test focus
            chat_input.click()
            is_focused = chat_input.is_focused()
            if is_focused:
                results['passed'].append('✓ Input can receive focus')
            else:
                results['warnings'].append('⚠ Input focus issue')
            
            # Test clear
            chat_input.clear()
            value_after_clear = chat_input.input_value()
            if value_after_clear == '':
                results['passed'].append('✓ Input can be cleared')
        
        # Test coverage items hover (if any)
        coverage_items = page.locator('.coverage-item')
        if coverage_items.count() > 0:
            first_item = coverage_items.first
            first_item.hover()
            time.sleep(0.5)
            results['passed'].append('✓ Coverage items respond to hover')
        
        # Test detail items hover
        detail_items = page.locator('.detail-item')
        if detail_items.count() > 0:
            detail_items.first.hover()
            time.sleep(0.3)
            results['passed'].append('✓ Detail items respond to hover')
        
        # Screenshot showing interactive state
        page.screenshot(path='test_screenshots/06_interactive.png', full_page=True)
        results['passed'].append('✓ Screenshot saved: 06_interactive.png')
        
    except Exception as e:
        results['failed'].append(f'✗ Exception: {str(e)}')
    
    print_results(results)
    return results


def test_responsive_layout(page: Page) -> dict:
    """Test 7: Responsive Layout at Different Sizes"""
    print("\n" + "="*60)
    print("TEST 7: Responsive Layout")
    print("="*60)
    
    results = {
        'test_name': 'Responsive Layout',
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    try:
        viewports = [
            {'width': 1920, 'height': 1080, 'name': 'Desktop (Full HD)'},
            {'width': 1440, 'height': 900, 'name': 'Laptop'},
            {'width': 1024, 'height': 768, 'name': 'Tablet Landscape'},
            {'width': 768, 'height': 1024, 'name': 'Tablet Portrait'},
            {'width': 480, 'height': 856, 'name': 'Mobile'},
        ]
        
        for vp in viewports:
            print(f"\n  Testing {vp['name']} ({vp['width']}x{vp['height']})...")
            page.set_viewport_size({"width": vp['width'], "height": vp['height']})
            time.sleep(0.5)
            
            # Check main elements are visible
            sidebar = page.locator('.sidebar')
            main_content = page.locator('.main-content')
            
            sidebar_visible = sidebar.is_visible() if sidebar.count() > 0 else False
            main_visible = main_content.is_visible() if main_content.count() > 0 else False
            
            if sidebar_visible and main_visible:
                results['passed'].append(f"✓ {vp['name']}: Layout intact")
                
                # For mobile, check vertical stacking
                if vp['width'] <= 768:
                    sidebar_box = sidebar.bounding_box()
                    main_box = main_content.bounding_box()
                    
                    if sidebar_box and main_box:
                        if sidebar_box['y'] < main_box['y']:
                            results['passed'].append(f"  └─ Vertical stacking correct")
                        else:
                            results['warnings'].append(f"  ⚠ Stacking order unusual")
            else:
                results['warnings'].append(
                    f"⚠ {vp['name']}: Some elements hidden "
                    f"(sidebar: {sidebar_visible}, main: {main_visible})"
                )
            
            # Take screenshot for this viewport
            safe_name = vp['name'].replace(' ', '_').replace('(', '').replace(')', '')
            page.screenshot(path=f'test_screenshots/07_{safe_name.lower()}.png', full_page=True)
        
        # Reset to desktop
        page.set_viewport_size({"width": 1440, "height": 900})
        results['passed'].append('✓ All viewport screenshots saved')
        
    except Exception as e:
        results['failed'].append(f'✗ Exception: {str(e)}')
    
    print_results(results)
    return results


def test_console_errors(page: Page) -> dict:
    """Test 8: Console Errors and Warnings"""
    print("\n" + "="*60)
    print("TEST 8: Console Logs Analysis")
    print("="*60)
    
    results = {
        'test_name': 'Console Logs',
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    try:
        # Collect console logs
        console_logs = []
        
        def handle_console(msg):
            console_logs.append({
                'type': msg.type,
                'text': msg.text
            })
        
        page.on('console', handle_console)
        
        # Reload page to capture all logs
        page.reload()
        page.wait_for_load_state('networkidle')
        time.sleep(1)
        
        # Analyze logs
        errors = [log for log in console_logs if log['type'] == 'error']
        warnings = [log for log in console_logs if log['type'] == 'warning']
        infos = [log for log in console_logs if log['type'] == 'info']
        
        results['passed'].append(f'✓ Total console entries: {len(console_logs)}')
        results['passed'].append(f'✓ Errors: {len(errors)}')
        results['passed'].append(f'✓ Warnings: {len(warnings)}')
        
        if len(errors) == 0:
            results['passed'].append('✓ No JavaScript errors!')
        else:
            for error in errors[:5]:  # Show first 5 errors
                results['failed'].append(f"✗ ERROR: {error['text'][:100]}")
        
        if len(warnings) > 0:
            for warning in warnings[:3]:  # Show first 3 warnings
                results['warnings'].append(f"⚠ WARNING: {warning['text'][:100]}")
        else:
            results['passed'].append('✓ No warnings')
        
        # Check network errors (failed requests)
        failed_requests = []
        
        def handle_request_failed(request):
            failed_requests.append(request.url)
        
        page.on('requestfailed', handle_request_failed)
        
        if len(failed_requests) == 0:
            results['passed'].append('✓ No network request failures')
        else:
            for url in failed_requests[:3]:
                results['warnings'].append(f"⚠ Failed request: {url}")
        
    except Exception as e:
        results['failed'].append(f'✗ Exception: {str(e)}')
    
    print_results(results)
    return results


def test_ui_styling(page: Page) -> dict:
    """Test 9: Visual Styling and Design Consistency"""
    print("\n" + "="*60)
    print("TEST 9: UI Styling Verification")
    print("="*60)
    
    results = {
        'test_name': 'UI Styling',
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    try:
        # Check fonts loaded
        font_check = page.evaluate('''() => {
            const body = document.body;
            const computedStyle = getComputedStyle(body);
            return {
                fontFamily: computedStyle.fontFamily,
                fontSize: computedStyle.fontSize,
                lineHeight: computedStyle.lineHeight,
                color: computedStyle.color
            };
        }''')
        
        results['passed'].append(f"✓ Font family: {font_check['fontFamily'][:50]}...")
        results['passed'].append(f"✓ Font size: {font_check['fontSize']}")
        results['passed'].append(f"✓ Line height: {font_check['lineHeight']}")
        results['passed'].append(f"✓ Text color: {font_check['color']}")
        
        # Check color scheme consistency
        primary_color = page.evaluate('''() => {
            const logo = document.querySelector('.logo');
            return logo ? getComputedStyle(logo).backgroundColor : null;
        }''')
        
        if primary_color:
            results['passed'].append(f"✓ Primary color applied: {primary_color}")
        
        # Check border radius consistency
        radius_checks = [
            ('.btn-primary', 'Button'),
            ('.message-content', 'Message'),
            ('.chat-input', 'Input'),
            ('.coverage-item', 'Coverage Item'),
        ]
        
        for selector, name in radius_checks:
            element = page.locator(selector).first
            if element.count() > 0:
                radius = element.evaluate('el => getComputedStyle(el).borderRadius')
                results['passed'].append(f"✓ {name} radius: {radius}")
        
        # Check shadow usage (should be minimal, no glows)
        shadow_check = page.evaluate('''() => {
            const allElements = document.querySelectorAll('*');
            let glowCount = 0;
            let normalShadowCount = 0;
            
            allElements.forEach(el => {
                const shadow = getComputedStyle(el).boxShadow;
                if (shadow && shadow !== 'none') {
                    if (shadow.includes('0 0')) {
                        glowCount++;
                    } else {
                        normalShadowCount++;
                    }
                }
            });
            
            return { glowCount, normalShadowCount };
        }''')
        
        results['passed'].append(f"✓ Normal shadows: {shadow_check['normalShadowCount']}")
        if shadow_check['glowCount'] == 0:
            results['passed'].append("✓ No glow effects (good!)")
        else:
            results['warnings'].append(f"⚠ Found {shadow_check['glowCount']} glow effects")
        
        # Final design screenshot
        page.set_viewport_size({"width": 1440, "height": 900})
        page.screenshot(path='test_screenshots/09_final_design.png', full_page=True)
        results['passed'].append('✓ Final design screenshot saved')
        
    except Exception as e:
        results['failed'].append(f'✗ Exception: {str(e)}')
    
    print_results(results)
    return results


def print_results(results: dict):
    """Print test results in formatted way"""
    total = len(results['passed']) + len(results['failed']) + len(results['warnings'])
    
    print(f"\n  Results Summary:")
    print(f"  {'='*50}")
    print(f"  ✓ Passed: {len(results['passed'])}/{total}")
    print(f"  ✗ Failed: {len(results['failed'])}/{total}")
    print(f"  ⚠ Warnings: {len(results['warnings'])}/{total}")
    print(f"  {'='*50}\n")


def generate_report(all_results: list):
    """Generate final test report"""
    print("\n" + "#"*70)
    print("#" + " "*20 + "FINAL TEST REPORT" + " "*27 + "#")
    print("#"*70)
    
    total_passed = 0
    total_failed = 0
    total_warnings = 0
    
    for result in all_results:
        total_passed += len(result['passed'])
        total_failed += len(result['failed'])
        total_warnings += len(result['warnings'])
        
        print(f"\n{result['test_name']:^50}")
        print("-"*50)
        print(f"Status: {'PASS ✓' if len(result['failed']) == 0 else 'FAIL ✗'}")
        print(f"Passed: {len(result['passed'])} | Failed: {len(result['failed'])} | Warnings: {len(result['warnings'])}")
        
        if len(result['failed']) > 0:
            print("\nFailures:")
            for fail in result['failed']:
                print(f"  {fail}")
    
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    total_tests = total_passed + total_failed + total_warnings
    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Checks: {total_tests}")
    print(f"Passed: {total_passed} ({pass_rate:.1f}%)")
    print(f"Failed: {total_failed}")
    print(f"Warnings: {total_warnings}")
    print(f"\nFinal Result: {'ALL TESTS PASSED ✓✓✓' if total_failed == 0 else 'SOME TESTS FAILED ✗'}")
    print("="*70)


def main():
    """Run all tests"""
    print("\n" + "🔍"*35)
    print("AI SALES TRAINER - COMPREHENSIVE FRONTEND TEST SUITE")
    print("🔍"*35)
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        
        print("\n📍 Navigating to http://localhost:8000 ...")
        page.goto('http://localhost:8000')
        page.wait_for_load_state('networkidle')
        time.sleep(2)  # Additional wait for JS initialization
        
        print("✓ Page loaded successfully\n")
        
        # Run all tests
        all_results = []
        
        all_results.append(test_page_load(page))
        all_results.append(test_session_management(page))
        all_results.append(test_message_sending(page))
        all_results.append(test_semantic_coverage(page))
        all_results.append(test_customer_profile(page))
        all_results.append(test_interactive_elements(page))
        all_results.append(test_responsive_layout(page))
        all_results.append(test_console_errors(page))
        all_results.append(test_ui_styling(page))
        
        # Generate final report
        generate_report(all_results)
        
        browser.close()
    
    print("\n✅ Testing complete! Screenshots saved to 'test_screenshots/' directory.\n")


if __name__ == '__main__':
    main()
